//! N-body Entropy CLI - Command-line interface for the n-body entropy generator

#[cfg(feature = "gpu")]
use nbody_entropy::GpuNbodyEntropy;
use nbody_entropy::NbodyEntropy;
use rand_core::{RngCore, SeedableRng};
use std::io::{self, Write};
use std::time::Instant;

/// Output raw bytes to stdout (for piping to dieharder/ent)
fn output_raw(count: usize) {
    let mut rng = NbodyEntropy::new();
    let mut stdout = io::stdout().lock();
    let mut buf = [0u8; 8192];

    let mut remaining = count;
    while remaining > 0 {
        let chunk = remaining.min(buf.len());
        rng.fill_bytes(&mut buf[..chunk]);
        if stdout.write_all(&buf[..chunk]).is_err() {
            break; // Pipe closed
        }
        remaining -= chunk;
    }
}

/// Output continuous stream of random bytes (for dieharder -g 200)
fn output_continuous() {
    let mut rng = NbodyEntropy::new();
    let mut stdout = io::stdout().lock();
    let mut buf = [0u8; 8192];

    loop {
        rng.fill_bytes(&mut buf);
        if stdout.write_all(&buf).is_err() {
            break; // Pipe closed
        }
    }
}

/// Output raw bytes from GPU to stdout (for piping to dieharder/ent)
#[cfg(feature = "gpu")]
fn gpu_output_raw(count: usize) {
    let mut rng = GpuNbodyEntropy::from_seed([42, 0, 0, 0, 0, 0, 0, 0]);
    let mut stdout = io::stdout().lock();
    let mut buf = [0u8; 8192];

    let mut remaining = count;
    while remaining > 0 {
        let chunk = remaining.min(buf.len());
        rng.fill_bytes(&mut buf[..chunk]);
        if stdout.write_all(&buf[..chunk]).is_err() {
            break; // Pipe closed
        }
        remaining -= chunk;
    }
}

/// Output continuous stream of GPU-generated random bytes
#[cfg(feature = "gpu")]
fn gpu_output_continuous() {
    let mut rng = GpuNbodyEntropy::from_seed([42, 0, 0, 0, 0, 0, 0, 0]);
    let mut stdout = io::stdout().lock();
    let mut buf = [0u8; 8192];

    loop {
        rng.fill_bytes(&mut buf);
        if stdout.write_all(&buf).is_err() {
            break; // Pipe closed
        }
    }
}

/// NIST SP 800-22 inspired test suite
fn run_nist_tests(sample_size: usize) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  NIST SP 800-22 Inspired Statistical Test Suite             ║");
    println!(
        "║  Sample size: {:>10} values ({:>6} MB)                  ║",
        sample_size,
        sample_size * 8 / 1_000_000
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut rng = NbodyEntropy::from_seed([42, 0, 0, 0, 0, 0, 0, 0]);
    let start = Instant::now();

    // Collect samples
    let mut samples: Vec<u64> = Vec::with_capacity(sample_size);
    for _ in 0..sample_size {
        samples.push(rng.next_u64());
    }
    let gen_time = start.elapsed();
    println!("Generated {} samples in {:?}\n", sample_size, gen_time);

    let mut passed = 0;
    let mut failed = 0;

    // Test 1: Frequency (Monobit) Test
    print!("1. Frequency (Monobit) Test... ");
    let total_bits = sample_size * 64;
    let ones: u64 = samples.iter().map(|x| x.count_ones() as u64).sum();
    let zeros = total_bits as u64 - ones;
    let s_obs = (ones as f64 - zeros as f64).abs() / (total_bits as f64).sqrt();
    let freq_pass = s_obs < 2.0;
    println!(
        "S_obs = {:.4} {}",
        s_obs,
        if freq_pass { "PASS" } else { "FAIL" }
    );
    if freq_pass {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 2: Block Frequency Test
    print!("2. Block Frequency Test (M=128)... ");
    let block_size = 128;
    let num_blocks = (sample_size * 64) / block_size;
    let mut chi_sq = 0.0;
    let mut bit_idx = 0;
    for _ in 0..num_blocks {
        let mut block_ones = 0u32;
        for _ in 0..block_size {
            let sample_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            if sample_idx < samples.len() && (samples[sample_idx] >> bit_pos) & 1 == 1 {
                block_ones += 1;
            }
            bit_idx += 1;
        }
        let pi = block_ones as f64 / block_size as f64;
        chi_sq += (pi - 0.5) * (pi - 0.5);
    }
    chi_sq *= 4.0 * block_size as f64;
    let block_freq_pass = chi_sq < num_blocks as f64 * 1.5;
    println!(
        "χ² = {:.2} (n={}) {}",
        chi_sq,
        num_blocks,
        if block_freq_pass { "PASS" } else { "FAIL" }
    );
    if block_freq_pass {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 3: Runs Test
    print!("3. Runs Test... ");
    let pi = ones as f64 / total_bits as f64;
    let tau = 2.0 / (total_bits as f64).sqrt();
    if (pi - 0.5).abs() >= tau {
        println!("SKIP (prerequisite failed: pi={:.4})", pi);
    } else {
        let mut runs = 1u64;
        let mut prev_bit = samples[0] & 1;
        for (idx, &sample) in samples.iter().enumerate() {
            let start_bit = if idx == 0 { 1 } else { 0 };
            for bit_pos in start_bit..64 {
                let curr_bit = (sample >> bit_pos) & 1;
                if curr_bit != prev_bit {
                    runs += 1;
                }
                prev_bit = curr_bit;
            }
        }
        let expected_runs = 2.0 * total_bits as f64 * pi * (1.0 - pi);
        let std_dev = (2.0 * (total_bits as f64).sqrt() * pi * (1.0 - pi)).abs();
        let z = if std_dev > 0.0 {
            (runs as f64 - expected_runs).abs() / std_dev
        } else {
            0.0
        };
        let runs_pass = z < 2.0;
        println!(
            "runs={}, z={:.4} {}",
            runs,
            z,
            if runs_pass { "PASS" } else { "FAIL" }
        );
        if runs_pass {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    // Test 4: Longest Run of Ones
    print!("4. Longest Run of Ones... ");
    let mut max_run = 0u32;
    let mut current_run = 0u32;
    for &sample in &samples {
        for bit_pos in 0..64 {
            if (sample >> bit_pos) & 1 == 1 {
                current_run += 1;
                max_run = max_run.max(current_run);
            } else {
                current_run = 0;
            }
        }
    }
    let expected_longest = (total_bits as f64).log2();
    let longest_pass = (max_run as f64) < expected_longest * 2.0;
    println!(
        "max={}, expected~{:.1} {}",
        max_run,
        expected_longest,
        if longest_pass { "PASS" } else { "FAIL" }
    );
    if longest_pass {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 5: Binary Matrix Rank Test
    print!("5. Binary Matrix Rank Test (32x32)... ");
    let matrices = sample_size / 16;
    let mut full_rank = 0u64;
    let mut rank_31 = 0u64;
    for m in 0..matrices.min(1000) {
        let base = m * 16;
        if base + 16 > samples.len() {
            break;
        }
        let mut matrix = [[0u8; 32]; 32];
        for row in 0..32 {
            let val_idx = base + row / 2;
            let val = samples[val_idx];
            let offset = if row % 2 == 0 { 0 } else { 32 };
            for col in 0..32 {
                matrix[row][col] = ((val >> (offset + col)) & 1) as u8;
            }
        }
        let rank = matrix_rank(&matrix);
        if rank == 32 {
            full_rank += 1;
        } else if rank == 31 {
            rank_31 += 1;
        }
    }
    let n = matrices.min(1000) as f64;
    let p_full = 0.2888;
    let p_31 = 0.5776;
    let expected_full = n * p_full;
    let expected_31 = n * p_31;
    let chi_sq = if n > 0.0 {
        (full_rank as f64 - expected_full).powi(2) / expected_full
            + (rank_31 as f64 - expected_31).powi(2) / expected_31
    } else {
        0.0
    };
    let rank_pass = chi_sq < 10.0;
    println!(
        "full={}, r31={}, χ²={:.2} {}",
        full_rank,
        rank_31,
        chi_sq,
        if rank_pass { "PASS" } else { "FAIL" }
    );
    if rank_pass {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 6: Spectral Test
    print!("6. Spectral Test (simplified)... ");
    let test_len = sample_size.min(10000);
    let mut max_autocorr = 0.0f64;
    for lag in [1, 2, 4, 8, 16, 32, 64, 128] {
        let mut corr = 0i64;
        for i in 0..(test_len - lag) {
            let a = samples[i].count_ones() as i64 - 32;
            let b = samples[i + lag].count_ones() as i64 - 32;
            corr += a * b;
        }
        let normalized = (corr as f64 / (test_len - lag) as f64).abs();
        max_autocorr = max_autocorr.max(normalized);
    }
    let spectral_pass = max_autocorr < 5.0;
    println!(
        "max_autocorr={:.4} {}",
        max_autocorr,
        if spectral_pass { "PASS" } else { "FAIL" }
    );
    if spectral_pass {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 7: Template Matching
    print!("7. Template Matching (pattern: 0xFF)... ");
    let mut matches = 0u64;
    for &sample in &samples {
        for shift in 0..57 {
            if (sample >> shift) & 0xFF == 0xFF {
                matches += 1;
            }
        }
    }
    let expected_matches = (sample_size * 57) as f64 / 256.0;
    let std_dev = (expected_matches * (1.0 - 1.0 / 256.0)).sqrt();
    let z = if std_dev > 0.0 {
        (matches as f64 - expected_matches).abs() / std_dev
    } else {
        0.0
    };
    let template_pass = z < 3.0;
    println!(
        "matches={}, expected={:.0}, z={:.2} {}",
        matches,
        expected_matches,
        z,
        if template_pass { "PASS" } else { "FAIL" }
    );
    if template_pass {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 8: Serial Test
    print!("8. Serial Test (2-bit patterns)... ");
    let mut pair_counts = [0u64; 4];
    for &sample in &samples {
        for i in 0..32 {
            let pair = ((sample >> (i * 2)) & 0b11) as usize;
            pair_counts[pair] += 1;
        }
    }
    let total_pairs = sample_size * 32;
    let expected_pair = total_pairs as f64 / 4.0;
    let chi_sq: f64 = pair_counts
        .iter()
        .map(|&c| (c as f64 - expected_pair).powi(2) / expected_pair)
        .sum();
    let serial_pass = chi_sq < 10.0;
    println!(
        "χ²={:.2} (counts: {:?}) {}",
        chi_sq,
        pair_counts,
        if serial_pass { "PASS" } else { "FAIL" }
    );
    if serial_pass {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 9: Approximate Entropy
    print!("9. Approximate Entropy Test... ");
    let m = 4;
    let mut counts_m = vec![0u64; 1 << m];
    let mut counts_m1 = vec![0u64; 1 << (m + 1)];
    for &sample in samples.iter().take(sample_size.min(50000)) {
        for shift in 0..(64 - m) {
            let pattern_m = ((sample >> shift) & ((1 << m) - 1)) as usize;
            counts_m[pattern_m] += 1;
        }
        for shift in 0..(64 - m - 1) {
            let pattern_m1 = ((sample >> shift) & ((1 << (m + 1)) - 1)) as usize;
            counts_m1[pattern_m1] += 1;
        }
    }
    let total_m: u64 = counts_m.iter().sum();
    let total_m1: u64 = counts_m1.iter().sum();
    let phi_m: f64 = counts_m
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total_m as f64;
            p * p.ln()
        })
        .sum();
    let phi_m1: f64 = counts_m1
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total_m1 as f64;
            p * p.ln()
        })
        .sum();
    let ap_en = phi_m - phi_m1;
    let entropy_pass = (ap_en - 0.693).abs() < 0.1;
    println!(
        "ApEn={:.4} (expected~0.693) {}",
        ap_en,
        if entropy_pass { "PASS" } else { "FAIL" }
    );
    if entropy_pass {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 10: Cumulative Sums
    print!("10. Cumulative Sums Test... ");
    let mut max_excursion = 0i64;
    let mut sum = 0i64;
    for &sample in samples.iter().take(sample_size.min(100000)) {
        for bit in 0..64 {
            sum += if (sample >> bit) & 1 == 1 { 1 } else { -1 };
            max_excursion = max_excursion.max(sum.abs());
        }
    }
    let n = (sample_size.min(100000) * 64) as f64;
    let expected_excursion = n.sqrt() * 1.5;
    let cusum_pass = (max_excursion as f64) < expected_excursion * 2.0;
    println!(
        "max={}, expected<{:.0} {}",
        max_excursion,
        expected_excursion * 2.0,
        if cusum_pass { "PASS" } else { "FAIL" }
    );
    if cusum_pass {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 11: Random Excursions
    print!("11. Random Excursions (zero crossings)... ");
    let mut crossings = 0u64;
    let mut sum = 0i64;
    let mut prev_sign = 0i64;
    for &sample in samples.iter().take(sample_size.min(100000)) {
        for bit in 0..64 {
            sum += if (sample >> bit) & 1 == 1 { 1 } else { -1 };
            let sign = sum.signum();
            if sign != prev_sign && prev_sign != 0 {
                crossings += 1;
            }
            prev_sign = sign;
        }
    }
    let n = (sample_size.min(100000) * 64) as f64;
    let expected_crossings = n.sqrt();
    let excursion_pass = (crossings as f64 - expected_crossings).abs() < expected_crossings;
    println!(
        "crossings={}, expected~{:.0} {}",
        crossings,
        expected_crossings,
        if excursion_pass { "PASS" } else { "FAIL" }
    );
    if excursion_pass {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 12: Monte Carlo Pi
    print!("12. Monte Carlo Pi Estimation... ");
    let mut inside = 0u64;
    let pairs = sample_size / 2;
    for i in 0..pairs {
        let x = (samples[i * 2] as f64) / (u64::MAX as f64);
        let y = (samples[i * 2 + 1] as f64) / (u64::MAX as f64);
        if x * x + y * y <= 1.0 {
            inside += 1;
        }
    }
    let pi_estimate = 4.0 * inside as f64 / pairs as f64;
    let pi_error = (pi_estimate - std::f64::consts::PI).abs();
    // Expected std dev for Monte Carlo Pi ≈ sqrt(π(4-π)/n) ≈ 0.007 for 50k pairs
    // Allow 3 sigma (~2.1%) to avoid false failures
    let pi_pass = pi_error < 0.025;
    println!(
        "π≈{:.6} (error={:.6}) {}",
        pi_estimate,
        pi_error,
        if pi_pass { "PASS" } else { "FAIL" }
    );
    if pi_pass {
        passed += 1;
    } else {
        failed += 1;
    }

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║  RESULTS: {}/{} tests passed                                   ║",
        passed,
        passed + failed
    );
    let status = if failed == 0 {
        "EXCELLENT"
    } else if failed <= 2 {
        "GOOD"
    } else if failed <= 4 {
        "ACCEPTABLE"
    } else {
        "POOR"
    };
    println!(
        "║  Overall: {:10}                                         ║",
        status
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
}

/// Compute rank of binary matrix using Gaussian elimination mod 2
fn matrix_rank(matrix: &[[u8; 32]; 32]) -> u32 {
    let mut m = *matrix;
    let mut rank = 0;

    for col in 0..32 {
        let mut pivot = None;
        for row in rank as usize..32 {
            if m[row][col] == 1 {
                pivot = Some(row);
                break;
            }
        }

        if let Some(pivot_row) = pivot {
            m.swap(rank as usize, pivot_row);
            for row in 0..32 {
                if row != rank as usize && m[row][col] == 1 {
                    for c in 0..32 {
                        m[row][c] ^= m[rank as usize][c];
                    }
                }
            }
            rank += 1;
        }
    }

    rank
}

fn benchmark() {
    println!("Benchmarking CPU generation speed...\n");

    let mut rng = NbodyEntropy::from_seed([12, 34, 56, 78, 90, 12, 34, 56]);
    let iterations = 1_000_000;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = rng.next_u64();
    }
    let elapsed = start.elapsed();

    let per_sec = iterations as f64 / elapsed.as_secs_f64();
    let mb_per_sec = per_sec * 8.0 / 1_000_000.0;

    println!("CPU: Generated {} u64 values in {:?}", iterations, elapsed);
    println!(
        "CPU Speed: {:.2} values/sec ({:.2} MB/s)",
        per_sec, mb_per_sec
    );
}

#[cfg(feature = "gpu")]
fn benchmark_gpu() {
    println!("Benchmarking GPU generation speed...\n");

    println!("Initializing GPU...");
    let mut rng = GpuNbodyEntropy::from_seed([12, 34, 56, 78, 90, 12, 34, 56]);
    println!("GPU initialized.\n");

    // Warm up
    for _ in 0..10 {
        let _ = rng.next_u64();
    }

    let iterations = 100_000; // Fewer iterations since GPU has overhead per call

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = rng.next_u64();
    }
    let elapsed = start.elapsed();

    let per_sec = iterations as f64 / elapsed.as_secs_f64();
    let mb_per_sec = per_sec * 8.0 / 1_000_000.0;

    println!("GPU: Generated {} u64 values in {:?}", iterations, elapsed);
    println!(
        "GPU Speed: {:.2} values/sec ({:.2} MB/s)",
        per_sec, mb_per_sec
    );
    println!(
        "\nNote: GPU version batches {} physics steps per query.",
        16
    );
    println!("Effective physics steps/sec: {:.2}", per_sec * 16.0);
}

#[cfg(feature = "gpu")]
fn run_gpu_tests(sample_size: usize) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  GPU-Accelerated NIST Test Suite                            ║");
    println!(
        "║  Sample size: {:>10} values ({:>6} MB)                  ║",
        sample_size,
        sample_size * 8 / 1_000_000
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Initializing GPU...");
    let mut rng = GpuNbodyEntropy::from_seed([42, 0, 0, 0, 0, 0, 0, 0]);
    println!("GPU initialized.\n");

    let start = Instant::now();

    // Collect samples
    let mut samples: Vec<u64> = Vec::with_capacity(sample_size);
    for _ in 0..sample_size {
        samples.push(rng.next_u64());
    }
    let gen_time = start.elapsed();
    println!("Generated {} samples in {:?}\n", sample_size, gen_time);

    // Run basic tests (reuse existing test logic)
    let mut passed = 0;
    let mut failed = 0;

    // Test 1: Frequency (Monobit) Test
    print!("1. Frequency (Monobit) Test... ");
    let total_bits = sample_size * 64;
    let ones: u64 = samples.iter().map(|x| x.count_ones() as u64).sum();
    let zeros = total_bits as u64 - ones;
    let s_obs = (ones as f64 - zeros as f64).abs() / (total_bits as f64).sqrt();
    let freq_pass = s_obs < 2.0;
    println!(
        "S_obs = {:.4} {}",
        s_obs,
        if freq_pass { "PASS" } else { "FAIL" }
    );
    if freq_pass {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 2: Bit distribution
    print!("2. Bit Distribution Test... ");
    let expected = sample_size as u64 * 32;
    let deviation = (ones as f64 - expected as f64).abs() / expected as f64;
    let bit_pass = deviation < 0.02;
    println!(
        "ones={}, expected={}, deviation={:.4} {}",
        ones,
        expected,
        deviation,
        if bit_pass { "PASS" } else { "FAIL" }
    );
    if bit_pass {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 3: Serial Test (2-bit patterns)
    print!("3. Serial Test (2-bit patterns)... ");
    let mut pair_counts = [0u64; 4];
    for &sample in &samples {
        for i in 0..32 {
            let pair = ((sample >> (i * 2)) & 0b11) as usize;
            pair_counts[pair] += 1;
        }
    }
    let total_pairs = sample_size * 32;
    let expected_pair = total_pairs as f64 / 4.0;
    let chi_sq: f64 = pair_counts
        .iter()
        .map(|&c| (c as f64 - expected_pair).powi(2) / expected_pair)
        .sum();
    let serial_pass = chi_sq < 10.0;
    println!(
        "χ²={:.2} {}",
        chi_sq,
        if serial_pass { "PASS" } else { "FAIL" }
    );
    if serial_pass {
        passed += 1;
    } else {
        failed += 1;
    }

    // Test 4: Monte Carlo Pi
    print!("4. Monte Carlo Pi Estimation... ");
    let mut inside = 0u64;
    let pairs = sample_size / 2;
    for i in 0..pairs {
        let x = (samples[i * 2] as f64) / (u64::MAX as f64);
        let y = (samples[i * 2 + 1] as f64) / (u64::MAX as f64);
        if x * x + y * y <= 1.0 {
            inside += 1;
        }
    }
    let pi_estimate = 4.0 * inside as f64 / pairs as f64;
    let pi_error = (pi_estimate - std::f64::consts::PI).abs();
    let pi_pass = pi_error < 0.025;
    println!(
        "π≈{:.6} (error={:.6}) {}",
        pi_estimate,
        pi_error,
        if pi_pass { "PASS" } else { "FAIL" }
    );
    if pi_pass {
        passed += 1;
    } else {
        failed += 1;
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║  RESULTS: {}/{} tests passed                                   ║",
        passed,
        passed + failed
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
}

fn print_usage() {
    eprintln!("nbody-entropy - Experimental entropy from N-body simulation\n");
    eprintln!("Usage: nbody-entropy [command]\n");
    eprintln!("Commands:");
    eprintln!("  test            Run built-in NIST-inspired test suite (default)");
    eprintln!("  raw <bytes>     Output N raw bytes to stdout");
    eprintln!("  stream          Output continuous stream (for dieharder -g 200)");
    eprintln!("  benchmark       Measure CPU generation speed");
    #[cfg(feature = "gpu")]
    {
        eprintln!("  gpu-test        Run GPU-accelerated test suite");
        eprintln!("  gpu-benchmark   Measure GPU generation speed");
        eprintln!("  gpu-raw <bytes> Output N raw GPU bytes to stdout");
        eprintln!("  gpu-stream      Output continuous GPU stream (for dieharder -g 200)");
    }
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("\nGPU commands available with: cargo run --features gpu");
    }
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  nbody-entropy test");
    eprintln!("  nbody-entropy raw 10000000 | ent");
    eprintln!("  nbody-entropy stream | dieharder -a -g 200");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        None | Some("test") => {
            run_nist_tests(100_000);
        }
        Some("raw") => {
            let bytes: usize = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1_000_000);
            output_raw(bytes);
        }
        Some("stream") => {
            output_continuous();
        }
        Some("benchmark") => {
            benchmark();
        }
        #[cfg(feature = "gpu")]
        Some("gpu-test") => {
            run_gpu_tests(50_000);
        }
        #[cfg(feature = "gpu")]
        Some("gpu-benchmark") => {
            benchmark_gpu();
        }
        #[cfg(feature = "gpu")]
        Some("gpu-raw") => {
            let bytes: usize = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1_000_000);
            gpu_output_raw(bytes);
        }
        #[cfg(feature = "gpu")]
        Some("gpu-stream") => {
            gpu_output_continuous();
        }
        Some("--help") | Some("-h") | Some("help") => {
            print_usage();
        }
        Some(other) => {
            eprintln!("Unknown command: {}", other);
            print_usage();
            std::process::exit(1);
        }
    }
}
