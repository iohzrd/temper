//! GPU Performance Benchmark
//!
//! Profiles the thermodynamic particle system at various scales to identify
//! bottlenecks and measure optimization improvements.

use nbody_entropy::thermodynamic::{LossFunction, ThermodynamicSystem};
use std::time::Instant;

fn main() {
    println!("{}",
        "╔══════════════════════════════════════════════════════════════════════════╗");
    println!("{}",
        "║           GPU PERFORMANCE BENCHMARK                                      ║");
    println!("{}",
        "╠══════════════════════════════════════════════════════════════════════════╣");
    println!("{}",
        "║  Profiling thermodynamic particle system with O(nK) optimization         ║");
    println!("{}",
        "╚══════════════════════════════════════════════════════════════════════════╝\n");

    let warmup_steps = 10;
    let bench_steps = 100;

    // Test 1: Compare different repulsion_samples settings
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("REPULSION SAMPLING COMPARISON (1000 particles, dim=4)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let repulsion_samples = [0, 32, 64, 128, 256, 512, 1000];
    println!("\n  {:>10} {:>12} {:>12} {:>12}", "Samples", "Steps/sec", "µs/step", "Speedup");
    println!("  {:->10} {:->12} {:->12} {:->12}", "", "", "", "");

    let baseline_time = benchmark_with_repulsion_samples(1000, 4, warmup_steps, bench_steps, LossFunction::Sphere, 1000);

    for &samples in &repulsion_samples {
        let us_per_step = benchmark_with_repulsion_samples(
            1000, 4, warmup_steps, bench_steps, LossFunction::Sphere, samples
        );
        let steps_per_sec = 1_000_000.0 / us_per_step;
        let speedup = baseline_time / us_per_step;
        let label = if samples == 0 { "skip".to_string() } else { samples.to_string() };
        println!("  {:>10} {:>12.1} {:>12.1} {:>12.1}x", label, steps_per_sec, us_per_step, speedup);
    }

    // Test 2: Scaling with particle count (using optimized K=64)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SCALING WITH PARTICLE COUNT (K=64 samples, dim=4)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let particle_counts = [100, 500, 1000, 2000, 4000, 8000, 10000, 16000];
    println!("\n  {:>8} {:>10} {:>12} {:>12} {:>10}", "Particles", "Steps/sec", "µs/step", "Per 10k", "Scaling");
    println!("  {:->8} {:->10} {:->12} {:->12} {:->10}", "", "", "", "", "");

    let mut first_time = 0.0;
    for &count in &particle_counts {
        let us_per_step = benchmark_with_repulsion_samples(
            count, 4, warmup_steps, bench_steps, LossFunction::Sphere, 64
        );
        let steps_per_sec = 1_000_000.0 / us_per_step;
        let per_10k_particles = us_per_step * 10000.0 / count as f64;
        let scaling = if first_time == 0.0 {
            first_time = us_per_step;
            1.0
        } else {
            us_per_step / first_time
        };
        println!("  {:>8} {:>10.1} {:>12.1} {:>12.1} {:>10.2}x", count, steps_per_sec, us_per_step, per_10k_particles, scaling);
    }

    // Test 3: Skip repulsion scaling (pure optimization mode)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SKIP REPULSION SCALING (K=0, pure optimization)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("\n  {:>8} {:>10} {:>12} {:>12} {:>10}", "Particles", "Steps/sec", "µs/step", "Per 10k", "Scaling");
    println!("  {:->8} {:->10} {:->12} {:->12} {:->10}", "", "", "", "", "");

    first_time = 0.0;
    for &count in &particle_counts {
        let us_per_step = benchmark_with_repulsion_samples(
            count, 4, warmup_steps, bench_steps, LossFunction::Sphere, 0
        );
        let steps_per_sec = 1_000_000.0 / us_per_step;
        let per_10k_particles = us_per_step * 10000.0 / count as f64;
        let scaling = if first_time == 0.0 {
            first_time = us_per_step;
            1.0
        } else {
            us_per_step / first_time
        };
        println!("  {:>8} {:>10.1} {:>12.1} {:>12.1} {:>10.2}x", count, steps_per_sec, us_per_step, per_10k_particles, scaling);
    }

    // Test 4: High-dimensional scaling with optimization
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("HIGH-DIM SCALING (4000 particles, K=64)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let dims = [4, 8, 16, 32, 37, 64];
    println!("\n  {:>6} {:>12} {:>12}", "Dim", "Steps/sec", "µs/step");
    println!("  {:->6} {:->12} {:->12}", "", "", "");

    for &dim in &dims {
        let us_per_step = benchmark_with_repulsion_samples(
            4000, dim, warmup_steps, bench_steps, LossFunction::Sphere, 64
        );
        let steps_per_sec = 1_000_000.0 / us_per_step;
        println!("  {:>6} {:>12.1} {:>12.1}", dim, steps_per_sec, us_per_step);
    }

    // Summary
    println!("\n{}",
        "╔══════════════════════════════════════════════════════════════════════════╗");
    println!("{}",
        "║                              SUMMARY                                     ║");
    println!("{}",
        "╠══════════════════════════════════════════════════════════════════════════╣");
    println!("{}",
        "║  OPTIMIZATION: Subsampled repulsion reduces O(n²) to O(nK)               ║");
    println!("{}",
        "║                                                                          ║");
    println!("{}",
        "║  MODES:                                                                  ║");
    println!("{}",
        "║  • K=0 (skip):  Fastest - for pure optimization (T→0)                    ║");
    println!("{}",
        "║  • K=64:        Default - good balance for sampling                      ║");
    println!("{}",
        "║  • K=N (full):  Most accurate - for precise SVGD                         ║");
    println!("{}",
        "╚══════════════════════════════════════════════════════════════════════════╝");
}

fn benchmark_with_repulsion_samples(
    particle_count: usize,
    dim: usize,
    warmup_steps: usize,
    bench_steps: usize,
    loss_fn: LossFunction,
    repulsion_samples: u32,
) -> f64 {
    let mut system = ThermodynamicSystem::with_loss_function(
        particle_count, dim, 0.1, loss_fn
    );
    system.set_repulsion_samples(repulsion_samples);

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

    elapsed.as_micros() as f64 / bench_steps as f64
}
