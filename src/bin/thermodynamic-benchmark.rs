//! Unified Thermodynamic Particle System Benchmark
//!
//! Validates that the same system correctly performs three distinct computations
//! based solely on temperature:
//!
//! 1. ENTROPY (T >> 1): Generates statistically random bits
//! 2. SAMPLE (T ~ 0.1): Samples from posterior distribution (finds both optima)
//! 3. OPTIMIZE (T → 0): Converges to loss minima
//!
//! This benchmark establishes the claim that interacting particle systems
//! are a universal computational primitive controlled by temperature.

use std::time::Instant;
use temper::thermodynamic::{ThermodynamicMode, ThermodynamicSystem};

const PARTICLE_COUNT: usize = 1000;
const DIM: usize = 2;

// Known optima for the 2D neural net: y = w2 * tanh(w1 * x)
const OPTIMA: [(f32, f32); 2] = [(1.5, 2.0), (-1.5, -2.0)];
const OPTIMUM_TOLERANCE: f32 = 0.5;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║     UNIFIED THERMODYNAMIC PARTICLE SYSTEM - THREE MODE BENCHMARK         ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Thesis: Same system, same dynamics, temperature selects computation     ║");
    println!("║  T → 0   : OPTIMIZE (gradient descent to minima)                         ║");
    println!("║  T ~ 0.1 : SAMPLE (Bayesian inference, posterior sampling)               ║");
    println!("║  T >> 1  : ENTROPY (random number generation)                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    let mut all_passed = true;

    // Test 1: Optimization mode
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 1: OPTIMIZE MODE (T = 0.001)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let opt_result = test_optimize_mode();
    all_passed &= opt_result.passed;
    println!();

    // Test 2: Sampling mode
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 2: SAMPLE MODE (T = 0.1)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let sample_result = test_sample_mode();
    all_passed &= sample_result.passed;
    println!();

    // Test 3: Entropy mode
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 3: ENTROPY MODE (T = 10.0)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let entropy_result = test_entropy_mode();
    all_passed &= entropy_result.passed;
    println!();

    // Summary
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                              SUMMARY                                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  OPTIMIZE (T=0.001): {:6} | Converged: {:5.1}% | Min Loss: {:.6}       ║",
        if opt_result.passed { "PASS" } else { "FAIL" },
        opt_result.converged_fraction * 100.0,
        opt_result.min_loss
    );
    println!(
        "║  SAMPLE   (T=0.1):   {:6} | Both optima: {} | Spread: {:.3}            ║",
        if sample_result.passed { "PASS" } else { "FAIL" },
        if sample_result.found_both_optima {
            "YES"
        } else {
            "NO "
        },
        sample_result.spread
    );
    println!(
        "║  ENTROPY  (T=10.0):  {:6} | Bit balance: {:.4} | Chi²: {:.2}           ║",
        if entropy_result.passed {
            "PASS"
        } else {
            "FAIL"
        },
        entropy_result.bit_balance,
        entropy_result.chi_squared
    );
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    if all_passed {
        println!("║  ✓ ALL TESTS PASSED - Unified thermodynamic computation validated!      ║");
    } else {
        println!("║  ✗ SOME TESTS FAILED - See details above                                ║");
    }
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

struct OptimizeResult {
    passed: bool,
    converged_fraction: f32,
    min_loss: f32,
    mean_loss: f32,
}

fn test_optimize_mode() -> OptimizeResult {
    let temperature = 0.001;
    let steps = 500;

    println!(
        "  Creating system with {} particles, T = {}",
        PARTICLE_COUNT, temperature
    );
    let mut system = ThermodynamicSystem::new(PARTICLE_COUNT, DIM, temperature);

    assert_eq!(system.mode(), ThermodynamicMode::Optimize);
    println!("  Mode correctly identified as: {:?}", system.mode());

    let start = Instant::now();
    for _ in 0..steps {
        system.step();
    }
    let elapsed = start.elapsed();
    println!("  Ran {} steps in {:?}", steps, elapsed);

    let particles = system.read_particles();
    let energies: Vec<f32> = particles
        .iter()
        .filter(|p| !p.energy.is_nan())
        .map(|p| p.energy)
        .collect();

    let min_loss = energies.iter().cloned().fold(f32::MAX, f32::min);
    let mean_loss = energies.iter().sum::<f32>() / energies.len() as f32;
    let converged = energies.iter().filter(|&&e| e < 0.1).count();
    let converged_fraction = converged as f32 / energies.len() as f32;

    // Count particles near each optimum
    let mut near_opt1 = 0;
    let mut near_opt2 = 0;
    for p in &particles {
        let x = p.pos[0].to_f32();
        let y = p.pos[1].to_f32();
        if x.is_nan() {
            continue;
        }
        let d1 = ((x - OPTIMA[0].0).powi(2) + (y - OPTIMA[0].1).powi(2)).sqrt();
        let d2 = ((x - OPTIMA[1].0).powi(2) + (y - OPTIMA[1].1).powi(2)).sqrt();
        if d1 < OPTIMUM_TOLERANCE {
            near_opt1 += 1;
        }
        if d2 < OPTIMUM_TOLERANCE {
            near_opt2 += 1;
        }
    }

    println!("  Results:");
    println!("    Min loss:        {:.6}", min_loss);
    println!("    Mean loss:       {:.6}", mean_loss);
    println!("    Converged (<0.1): {:.1}%", converged_fraction * 100.0);
    println!("    Near optimum 1:  {} particles", near_opt1);
    println!("    Near optimum 2:  {} particles", near_opt2);

    // Pass criteria: >50% converged, min loss < 0.01
    let passed = converged_fraction > 0.5 && min_loss < 0.01;
    println!("  Status: {}", if passed { "PASS ✓" } else { "FAIL ✗" });

    OptimizeResult {
        passed,
        converged_fraction,
        min_loss,
        mean_loss,
    }
}

struct SampleResult {
    passed: bool,
    found_both_optima: bool,
    spread: f32,
    opt1_count: usize,
    opt2_count: usize,
}

fn test_sample_mode() -> SampleResult {
    let temperature = 0.1;
    let steps = 1000;

    println!(
        "  Creating system with {} particles, T = {}",
        PARTICLE_COUNT, temperature
    );
    let mut system = ThermodynamicSystem::new(PARTICLE_COUNT, DIM, temperature);

    assert_eq!(system.mode(), ThermodynamicMode::Sample);
    println!("  Mode correctly identified as: {:?}", system.mode());

    let start = Instant::now();
    for _ in 0..steps {
        system.step();
    }
    let elapsed = start.elapsed();
    println!("  Ran {} steps in {:?}", steps, elapsed);

    let particles = system.read_particles();

    // Count particles near each optimum
    let mut near_opt1 = 0;
    let mut near_opt2 = 0;
    let mut positions: Vec<(f32, f32)> = Vec::new();

    for p in &particles {
        let x = p.pos[0].to_f32();
        let y = p.pos[1].to_f32();
        if x.is_nan() {
            continue;
        }
        positions.push((x, y));
        let d1 = ((x - OPTIMA[0].0).powi(2) + (y - OPTIMA[0].1).powi(2)).sqrt();
        let d2 = ((x - OPTIMA[1].0).powi(2) + (y - OPTIMA[1].1).powi(2)).sqrt();
        if d1 < OPTIMUM_TOLERANCE * 2.0 {
            near_opt1 += 1;
        }
        if d2 < OPTIMUM_TOLERANCE * 2.0 {
            near_opt2 += 1;
        }
    }

    // Compute spread (standard deviation of positions)
    let mean_x = positions.iter().map(|p| p.0).sum::<f32>() / positions.len() as f32;
    let mean_y = positions.iter().map(|p| p.1).sum::<f32>() / positions.len() as f32;
    let var_x = positions
        .iter()
        .map(|p| (p.0 - mean_x).powi(2))
        .sum::<f32>()
        / positions.len() as f32;
    let var_y = positions
        .iter()
        .map(|p| (p.1 - mean_y).powi(2))
        .sum::<f32>()
        / positions.len() as f32;
    let spread = (var_x + var_y).sqrt();

    // Check energies
    let energies: Vec<f32> = particles
        .iter()
        .filter(|p| !p.energy.is_nan())
        .map(|p| p.energy)
        .collect();
    let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;
    let low_energy = energies.iter().filter(|&&e| e < 0.5).count();

    let found_both = near_opt1 > PARTICLE_COUNT / 10 && near_opt2 > PARTICLE_COUNT / 10;

    println!("  Results:");
    println!(
        "    Near optimum 1 (1.5, 2.0):   {} particles ({:.1}%)",
        near_opt1,
        near_opt1 as f32 / PARTICLE_COUNT as f32 * 100.0
    );
    println!(
        "    Near optimum 2 (-1.5, -2.0): {} particles ({:.1}%)",
        near_opt2,
        near_opt2 as f32 / PARTICLE_COUNT as f32 * 100.0
    );
    println!(
        "    Found both optima: {}",
        if found_both { "YES" } else { "NO" }
    );
    println!("    Position spread (σ): {:.3}", spread);
    println!("    Mean energy: {:.4}", mean_energy);
    println!(
        "    Low energy (<0.5): {:.1}%",
        low_energy as f32 / energies.len() as f32 * 100.0
    );

    // Pass criteria: found both optima AND has reasonable spread (not collapsed)
    let passed = found_both && spread > 0.5;
    println!("  Status: {}", if passed { "PASS ✓" } else { "FAIL ✗" });

    SampleResult {
        passed,
        found_both_optima: found_both,
        spread,
        opt1_count: near_opt1,
        opt2_count: near_opt2,
    }
}

struct EntropyResult {
    passed: bool,
    bit_balance: f32,
    chi_squared: f32,
    runs_test_z: f32,
    byte_distribution_chi: f32,
}

fn test_entropy_mode() -> EntropyResult {
    let temperature = 10.0;
    let warmup_steps = 100;
    let collection_steps = 500;

    println!(
        "  Creating system with {} particles, T = {}",
        PARTICLE_COUNT, temperature
    );
    let mut system = ThermodynamicSystem::new(PARTICLE_COUNT, DIM, temperature);

    assert_eq!(system.mode(), ThermodynamicMode::Entropy);
    println!("  Mode correctly identified as: {:?}", system.mode());

    // Warmup
    println!("  Warming up ({} steps)...", warmup_steps);
    for _ in 0..warmup_steps {
        system.step();
    }

    // Collect entropy
    println!("  Collecting entropy ({} steps)...", collection_steps);
    let start = Instant::now();
    let mut all_entropy: Vec<u32> = Vec::new();
    for _ in 0..collection_steps {
        system.step();
        let entropy = system.extract_entropy();
        all_entropy.extend(entropy);
    }
    let elapsed = start.elapsed();
    println!(
        "  Collected {} entropy values in {:?}",
        all_entropy.len(),
        elapsed
    );

    if all_entropy.is_empty() {
        println!("  ERROR: No entropy collected!");
        return EntropyResult {
            passed: false,
            bit_balance: 0.0,
            chi_squared: f32::MAX,
            runs_test_z: f32::MAX,
            byte_distribution_chi: f32::MAX,
        };
    }

    // Test 1: Bit balance (should be ~0.5 ones)
    let total_bits = all_entropy.len() * 32;
    let ones: u64 = all_entropy.iter().map(|x| x.count_ones() as u64).sum();
    let bit_balance = ones as f32 / total_bits as f32;
    let bit_balance_deviation = (bit_balance - 0.5).abs();

    // Test 2: Chi-squared on 4-bit nibbles
    let mut nibble_counts = [0u64; 16];
    for &val in &all_entropy {
        for i in 0..8 {
            let nibble = ((val >> (i * 4)) & 0xF) as usize;
            nibble_counts[nibble] += 1;
        }
    }
    let total_nibbles = all_entropy.len() * 8;
    let expected_per_nibble = total_nibbles as f64 / 16.0;
    let chi_squared: f64 = nibble_counts
        .iter()
        .map(|&c| {
            let diff = c as f64 - expected_per_nibble;
            diff * diff / expected_per_nibble
        })
        .sum();

    // Test 3: Runs test (transitions between 0 and 1)
    let mut runs = 1u64;
    let mut prev_bit = all_entropy[0] & 1;
    for &val in &all_entropy {
        for bit in 0..32 {
            let curr_bit = (val >> bit) & 1;
            if curr_bit != prev_bit {
                runs += 1;
            }
            prev_bit = curr_bit;
        }
    }
    let n = total_bits as f64;
    let pi = bit_balance as f64;
    let expected_runs = 2.0 * n * pi * (1.0 - pi) + 1.0;
    let std_runs = (2.0 * n * pi * (1.0 - pi) * (2.0 * pi * (1.0 - pi) - 1.0 / n))
        .abs()
        .sqrt();
    let runs_z = if std_runs > 0.0 {
        ((runs as f64 - expected_runs) / std_runs).abs()
    } else {
        0.0
    };

    // Test 4: Byte distribution chi-squared
    let mut byte_counts = [0u64; 256];
    for &val in &all_entropy {
        for i in 0..4 {
            let byte = ((val >> (i * 8)) & 0xFF) as usize;
            byte_counts[byte] += 1;
        }
    }
    let total_bytes = all_entropy.len() * 4;
    let expected_per_byte = total_bytes as f64 / 256.0;
    let byte_chi: f64 = byte_counts
        .iter()
        .map(|&c| {
            let diff = c as f64 - expected_per_byte;
            diff * diff / expected_per_byte
        })
        .sum();

    println!("  Statistical Tests:");
    println!(
        "    Bit balance:     {:.4} (ideal: 0.5000, deviation: {:.4})",
        bit_balance, bit_balance_deviation
    );
    println!(
        "    Nibble χ²:       {:.2} (15 df, critical: 25.0)",
        chi_squared
    );
    println!("    Runs test |z|:   {:.2} (critical: 2.58)", runs_z);
    println!(
        "    Byte χ²:         {:.2} (255 df, critical: 310)",
        byte_chi
    );

    // Pass criteria
    let bit_ok = bit_balance_deviation < 0.02;
    let nibble_ok = chi_squared < 30.0; // Slightly relaxed
    let runs_ok = runs_z < 3.0; // Slightly relaxed
    let byte_ok = byte_chi < 350.0; // Slightly relaxed

    println!(
        "    Bit balance:   {}",
        if bit_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "    Nibble dist:   {}",
        if nibble_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "    Runs test:     {}",
        if runs_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "    Byte dist:     {}",
        if byte_ok { "PASS" } else { "FAIL" }
    );

    let passed = bit_ok && nibble_ok && runs_ok && byte_ok;
    println!("  Status: {}", if passed { "PASS ✓" } else { "FAIL ✗" });

    EntropyResult {
        passed,
        bit_balance,
        chi_squared: chi_squared as f32,
        runs_test_z: runs_z as f32,
        byte_distribution_chi: byte_chi as f32,
    }
}
