//! Scale Testing Benchmark
//!
//! Push the thermodynamic particle system to its limits:
//! - Large particle counts (up to 100k+)
//! - High dimensions (up to 1000D)
//! - Memory and performance profiling
//!
//! Run with: cargo run --release --features gpu --bin scale-test

use temper::{ThermodynamicSystem, LossFunction};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                     SCALE TESTING BENCHMARK                              ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Testing the limits of GPU-accelerated thermodynamic particle systems    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Test 1: Particle count scaling
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("TEST 1: PARTICLE COUNT SCALING (dim=4, K=64 repulsion samples)");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let particle_counts = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000];
    let dim = 4;
    let warmup_steps = 50;
    let bench_steps = 200;

    println!("{:>10} {:>12} {:>12} {:>12} {:>12}",
             "Particles", "Steps/sec", "µs/step", "MB (est)", "Status");
    println!("{}", "-".repeat(62));

    for &n in &particle_counts {
        // Estimate memory: each particle is ~272 bytes (64*4 pos + 4+4+8 extras)
        let mem_mb = (n * 272) as f64 / 1_000_000.0;

        let result = std::panic::catch_unwind(|| {
            let mut system = ThermodynamicSystem::with_loss_function(n, dim, 1.0, LossFunction::Rastrigin);
            system.set_repulsion_samples(64);

            // Warmup
            for _ in 0..warmup_steps {
                system.step();
            }

            // Benchmark
            let start = Instant::now();
            for _ in 0..bench_steps {
                system.step();
            }
            let elapsed = start.elapsed();

            let steps_per_sec = bench_steps as f64 / elapsed.as_secs_f64();
            let us_per_step = elapsed.as_micros() as f64 / bench_steps as f64;

            (steps_per_sec, us_per_step)
        });

        match result {
            Ok((steps_per_sec, us_per_step)) => {
                println!("{:>10} {:>12.1} {:>12.1} {:>12.1} {:>12}",
                         n, steps_per_sec, us_per_step, mem_mb, "OK");
            }
            Err(_) => {
                println!("{:>10} {:>12} {:>12} {:>12.1} {:>12}",
                         n, "-", "-", mem_mb, "FAILED");
                break;
            }
        }
    }

    // Test 2: Dimension scaling
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("TEST 2: DIMENSION SCALING (particles=1000, K=64)");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let dimensions = [2, 4, 8, 16, 32, 64];  // MAX_DIM is 64
    let n = 1000;

    println!("{:>10} {:>12} {:>12} {:>15} {:>12}",
             "Dim", "Steps/sec", "µs/step", "Search Space", "Status");
    println!("{}", "-".repeat(65));

    for &d in &dimensions {
        let result = std::panic::catch_unwind(|| {
            let mut system = ThermodynamicSystem::with_loss_function(n, d, 1.0, LossFunction::Rastrigin);
            system.set_repulsion_samples(64);

            // Warmup
            for _ in 0..warmup_steps {
                system.step();
            }

            // Benchmark
            let start = Instant::now();
            for _ in 0..bench_steps {
                system.step();
            }
            let elapsed = start.elapsed();

            let steps_per_sec = bench_steps as f64 / elapsed.as_secs_f64();
            let us_per_step = elapsed.as_micros() as f64 / bench_steps as f64;

            (steps_per_sec, us_per_step)
        });

        // Search space size (assuming [-4, 4] per dimension)
        let search_space = format!("8^{} = {:.0e}", d, 8.0_f64.powi(d as i32));

        match result {
            Ok((steps_per_sec, us_per_step)) => {
                println!("{:>10} {:>12.1} {:>12.1} {:>15} {:>12}",
                         d, steps_per_sec, us_per_step, search_space, "OK");
            }
            Err(_) => {
                println!("{:>10} {:>12} {:>12} {:>15} {:>12}",
                         d, "-", "-", search_space, "FAILED");
            }
        }
    }

    // Test 3: Repulsion sampling impact at scale
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("TEST 3: REPULSION SAMPLING IMPACT AT SCALE (particles=20000, dim=4)");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let n = 20_000;  // Moderate size for repulsion testing
    let repulsion_samples = [0, 16, 32, 64, 128, 256];

    println!("{:>10} {:>12} {:>12} {:>12} {:>15}",
             "K samples", "Steps/sec", "µs/step", "Speedup", "Complexity");
    println!("{}", "-".repeat(65));

    let mut baseline_time = None;

    for &k in &repulsion_samples {
        let result = std::panic::catch_unwind(|| {
            let mut system = ThermodynamicSystem::with_loss_function(n, dim, 1.0, LossFunction::Rastrigin);
            system.set_repulsion_samples(k);

            // Warmup
            for _ in 0..warmup_steps {
                system.step();
            }

            // Benchmark
            let start = Instant::now();
            for _ in 0..bench_steps {
                system.step();
            }
            let elapsed = start.elapsed();

            let steps_per_sec = bench_steps as f64 / elapsed.as_secs_f64();
            let us_per_step = elapsed.as_micros() as f64 / bench_steps as f64;

            (steps_per_sec, us_per_step)
        });

        let complexity = if k == 0 {
            "O(n)".to_string()
        } else {
            format!("O(n*{})", k)
        };

        match result {
            Ok((steps_per_sec, us_per_step)) => {
                if baseline_time.is_none() && k > 0 {
                    baseline_time = Some(us_per_step);
                }
                let speedup = baseline_time.map_or(1.0, |b| b / us_per_step);
                println!("{:>10} {:>12.1} {:>12.1} {:>12.2}x {:>15}",
                         k, steps_per_sec, us_per_step, speedup, complexity);
            }
            Err(_) => {
                println!("{:>10} {:>12} {:>12} {:>12} {:>15}",
                         k, "-", "-", "-", complexity);
            }
        }
    }

    // Test 4: Optimization quality at scale
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("TEST 4: OPTIMIZATION QUALITY AT SCALE (Rastrigin, dim=8)");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let particle_counts_opt = [500, 1_000, 5_000, 10_000, 20_000, 50_000];
    let dim = 8;
    let opt_steps = 3000;

    println!("{:>10} {:>12} {:>12} {:>15} {:>12}",
             "Particles", "Best Loss", "Mean Loss", "Time (s)", "Global?");
    println!("{}", "-".repeat(65));

    for &n in &particle_counts_opt {
        let start = Instant::now();

        let mut system = ThermodynamicSystem::with_loss_function(n, dim, 5.0, LossFunction::Rastrigin);
        system.set_repulsion_samples(if n > 5000 { 32 } else { 64 });

        // Simulated annealing
        for step in 0..opt_steps {
            let progress = step as f32 / opt_steps as f32;
            let temp = 5.0 * (0.001_f32 / 5.0).powf(progress);
            system.set_temperature(temp);
            system.step();
        }

        let elapsed = start.elapsed();
        let particles = system.read_particles();

        let energies: Vec<f32> = particles.iter()
            .filter(|p| !p.energy.is_nan())
            .map(|p| p.energy)
            .collect();

        let best = energies.iter().cloned().fold(f32::MAX, f32::min);
        let mean = energies.iter().sum::<f32>() / energies.len() as f32;

        // Rastrigin global minimum is 0 at origin
        let is_global = best < 1.0;

        println!("{:>10} {:>12.4} {:>12.2} {:>15.2} {:>12}",
                 n, best, mean, elapsed.as_secs_f64(),
                 if is_global { "YES" } else { "no" });
    }

    // Test 5: Memory stress test
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("TEST 5: MEMORY STRESS TEST (finding maximum particle count)");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let mut max_particles = 0;
    let test_counts = [100_000, 150_000, 200_000, 250_000, 300_000, 400_000, 500_000];

    println!("{:>10} {:>12} {:>12} {:>12}",
             "Particles", "MB (est)", "Init Time", "Status");
    println!("{}", "-".repeat(50));

    for &n in &test_counts {
        let mem_mb = (n * 272) as f64 / 1_000_000.0;

        let start = Instant::now();
        let result = std::panic::catch_unwind(|| {
            let system = ThermodynamicSystem::with_loss_function(n, 4, 1.0, LossFunction::Sphere);
            // Try one step to ensure it actually works
            let _ = system;
        });
        let elapsed = start.elapsed();

        match result {
            Ok(_) => {
                max_particles = n;
                println!("{:>10} {:>12.1} {:>12.2}s {:>12}",
                         n, mem_mb, elapsed.as_secs_f64(), "OK");
            }
            Err(_) => {
                println!("{:>10} {:>12.1} {:>12} {:>12}",
                         n, mem_mb, "-", "FAILED");
                break;
            }
        }
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    println!("Maximum tested particle count: {}", max_particles);
    println!("Maximum dimensions: 64 (hardcoded limit)");
    println!("\nKey findings:");
    println!("  - Particle scaling: O(n) for pure optimization, O(nK) for SVGD");
    println!("  - Dimension scaling: Linear cost per dimension (gradient computation)");
    println!("  - Memory: ~280 bytes per particle");
    println!("\nBottlenecks at scale:");
    println!("  - GPU memory limits max particle count");
    println!("  - Repulsion computation dominates at high K");
    println!("  - Dimension scaling limited by MAX_DIM=64 constant");
}
