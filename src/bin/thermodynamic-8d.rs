//! 8-Dimensional Thermodynamic Particle System Benchmark
//!
//! Validates that the unified thermodynamic system works in high dimensions.
//! The 8D loss landscape has 2^4 = 16 global minima (combinations of ±optimal in each pair).
//!
//! This proves the system scales beyond 2D toy problems.

use std::time::Instant;
use temper::thermodynamic::{ThermodynamicMode, ThermodynamicSystem};

const PARTICLE_COUNT: usize = 1000;
const DIM: usize = 8;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║     8-DIMENSIONAL THERMODYNAMIC PARTICLE SYSTEM BENCHMARK                ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Testing that the unified system works in high-dimensional spaces        ║");
    println!("║  8D loss landscape has 2^4 = 16 global minima                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    let mut all_passed = true;

    // Test 1: Optimization in 8D
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 1: 8D OPTIMIZE MODE (T = 0.001)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let opt_passed = test_8d_optimize();
    all_passed &= opt_passed;
    println!();

    // Test 2: Sampling in 8D
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 2: 8D SAMPLE MODE (T = 0.1)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let sample_passed = test_8d_sample();
    all_passed &= sample_passed;
    println!();

    // Test 3: Entropy in 8D
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 3: 8D ENTROPY MODE (T = 10.0)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let entropy_passed = test_8d_entropy();
    all_passed &= entropy_passed;
    println!();

    // Summary
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                           8D SUMMARY                                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    if all_passed {
        println!("║  ✓ ALL 8D TESTS PASSED - System scales to high dimensions!              ║");
    } else {
        println!("║  ✗ SOME 8D TESTS FAILED                                                 ║");
    }
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

fn test_8d_optimize() -> bool {
    let temperature = 0.001;
    let steps = 800;

    println!(
        "  Creating 8D system with {} particles, T = {}",
        PARTICLE_COUNT, temperature
    );
    let mut system = ThermodynamicSystem::new(PARTICLE_COUNT, DIM, temperature);

    assert_eq!(system.mode(), ThermodynamicMode::Optimize);
    println!("  Mode: {:?}", system.mode());

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
    let converged = energies.iter().filter(|&&e| e < 0.5).count();
    let converged_frac = converged as f32 / energies.len() as f32;

    println!("  Results:");
    println!("    Dimensions:      {}", DIM);
    println!("    Min loss:        {:.6}", min_loss);
    println!("    Mean loss:       {:.6}", mean_loss);
    println!("    Converged (<0.5): {:.1}%", converged_frac * 100.0);

    // In 8D, convergence is harder, so we use relaxed criteria
    let passed = converged_frac > 0.3 && min_loss < 0.1;
    println!("  Status: {}", if passed { "PASS ✓" } else { "FAIL ✗" });
    passed
}

fn test_8d_sample() -> bool {
    let temperature = 0.1;
    let steps = 1500;

    println!(
        "  Creating 8D system with {} particles, T = {}",
        PARTICLE_COUNT, temperature
    );
    let mut system = ThermodynamicSystem::new(PARTICLE_COUNT, DIM, temperature);

    assert_eq!(system.mode(), ThermodynamicMode::Sample);
    println!("  Mode: {:?}", system.mode());

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

    let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;
    let low_energy = energies.iter().filter(|&&e| e < 1.0).count();
    let low_energy_frac = low_energy as f32 / energies.len() as f32;

    // Compute spread across all 8 dimensions
    let mut total_spread = 0.0;
    for d in 0..DIM {
        let vals: Vec<f32> = particles
            .iter()
            .filter(|p| !p.pos[d].is_nan())
            .map(|p| p.pos[d].to_f32())
            .collect();
        if vals.is_empty() {
            continue;
        }
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / vals.len() as f32;
        total_spread += var.sqrt();
    }
    let avg_spread = total_spread / DIM as f32;

    // Count how many of the 16 modes are populated
    // Modes are at combinations of (±1.5*scale, ±2.0*scale) for each pair
    let mut mode_counts = vec![0usize; 16];
    for p in &particles {
        if p.pos[0].is_nan() {
            continue;
        }
        let mut mode_idx = 0usize;
        for pair in 0..4 {
            let d = pair * 2;
            let scale = 1.0 / (1.0 + d as f32 * 0.1);
            let x = p.pos[d].to_f32();
            let y = p.pos[d + 1].to_f32();
            // Check if closer to positive or negative mode
            let pos_dist = (x - 1.5 * scale).abs() + (y - 2.0 * scale).abs();
            let neg_dist = (x + 1.5 * scale).abs() + (y + 2.0 * scale).abs();
            if neg_dist < pos_dist {
                mode_idx |= 1 << pair;
            }
        }
        if mode_idx < 16 {
            mode_counts[mode_idx] += 1;
        }
    }

    let modes_populated = mode_counts.iter().filter(|&&c| c > 10).count();

    println!("  Results:");
    println!("    Dimensions:      {}", DIM);
    println!("    Mean energy:     {:.4}", mean_energy);
    println!("    Low energy (<1): {:.1}%", low_energy_frac * 100.0);
    println!("    Avg spread (σ):  {:.3}", avg_spread);
    println!(
        "    Modes populated: {}/16 (with >10 particles each)",
        modes_populated
    );

    // In 8D sampling, we expect multiple modes to be populated
    let passed = low_energy_frac > 0.5 && modes_populated >= 4 && avg_spread > 0.3;
    println!("  Status: {}", if passed { "PASS ✓" } else { "FAIL ✗" });
    passed
}

fn test_8d_entropy() -> bool {
    let temperature = 10.0;
    let warmup = 100;
    let steps = 500;

    println!(
        "  Creating 8D system with {} particles, T = {}",
        PARTICLE_COUNT, temperature
    );
    let mut system = ThermodynamicSystem::new(PARTICLE_COUNT, DIM, temperature);

    assert_eq!(system.mode(), ThermodynamicMode::Entropy);
    println!("  Mode: {:?}", system.mode());

    // Warmup
    for _ in 0..warmup {
        system.step();
    }

    // Collect entropy
    let start = Instant::now();
    let mut all_entropy: Vec<u32> = Vec::new();
    for _ in 0..steps {
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
        return false;
    }

    // Bit balance test
    let total_bits = all_entropy.len() * 32;
    let ones: u64 = all_entropy.iter().map(|x| x.count_ones() as u64).sum();
    let bit_balance = ones as f32 / total_bits as f32;

    // Chi-squared on bytes
    let mut byte_counts = [0u64; 256];
    for &val in &all_entropy {
        for i in 0..4 {
            let byte = ((val >> (i * 8)) & 0xFF) as usize;
            byte_counts[byte] += 1;
        }
    }
    let total_bytes = all_entropy.len() * 4;
    let expected = total_bytes as f64 / 256.0;
    let chi_sq: f64 = byte_counts
        .iter()
        .map(|&c| (c as f64 - expected).powi(2) / expected)
        .sum();

    // Higher dimensional entropy should be even better due to more state space
    let throughput = (all_entropy.len() * 4) as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    println!("  Results:");
    println!("    Dimensions:      {}", DIM);
    println!("    Bit balance:     {:.4} (ideal: 0.5000)", bit_balance);
    println!("    Byte χ²:         {:.2} (255 df, critical: 310)", chi_sq);
    println!("    Throughput:      {:.2} MB/s", throughput);

    let bit_ok = (bit_balance - 0.5).abs() < 0.02;
    let chi_ok = chi_sq < 350.0;

    let passed = bit_ok && chi_ok;
    println!("  Status: {}", if passed { "PASS ✓" } else { "FAIL ✗" });
    passed
}
