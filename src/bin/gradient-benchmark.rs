//! Gradient Benchmark: Numerical vs Analytical
//!
//! Compares performance and accuracy of numerical finite-difference gradients
//! vs symbolic analytical gradients for the Expression DSL.
//!
//! Run with: cargo run --release --features gpu --bin gradient-benchmark

use temper::expr::*;
use temper::ThermodynamicSystem;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║               GRADIENT BENCHMARK: Analytical vs Numerical               ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Comparing symbolic differentiation vs finite differences on GPU        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Test functions
    let functions: Vec<(&str, Box<dyn Fn() -> Expr>)> = vec![
        ("Sphere", Box::new(|| sphere())),
        ("Rastrigin", Box::new(|| rastrigin())),
        ("Griewank", Box::new(|| griewank())),
        ("Levy", Box::new(|| levy())),
        ("Ackley", Box::new(|| ackley())),
    ];

    let particle_count = 2000;
    let dim = 8;
    let warmup_steps = 50;
    let bench_steps = 500;

    println!("Configuration:");
    println!("  Particles: {}", particle_count);
    println!("  Dimensions: {}", dim);
    println!("  Warmup steps: {}", warmup_steps);
    println!("  Benchmark steps: {}", bench_steps);
    println!();

    println!("{:>12} {:>14} {:>14} {:>12} {:>12}",
             "Function", "Analytical", "Numerical", "Speedup", "A.Best/N.Best");
    println!("{}", "-".repeat(70));

    for (name, expr_fn) in &functions {
        // Test analytical gradients
        let expr_analytical = expr_fn();
        let analytical_result = benchmark_expr(
            expr_analytical,
            true, // analytical
            particle_count,
            dim,
            warmup_steps,
            bench_steps,
        );

        // Test numerical gradients
        let expr_numerical = expr_fn();
        let numerical_result = benchmark_expr(
            expr_numerical,
            false, // numerical
            particle_count,
            dim,
            warmup_steps,
            bench_steps,
        );

        let speedup = numerical_result.us_per_step / analytical_result.us_per_step;
        let quality_ratio = analytical_result.best_energy / numerical_result.best_energy;

        println!("{:>12} {:>12.1}µs {:>12.1}µs {:>11.2}x {:>12.4}",
                 name,
                 analytical_result.us_per_step,
                 numerical_result.us_per_step,
                 speedup,
                 quality_ratio);
    }

    println!();
    println!("Legend:");
    println!("  Speedup > 1.0 means analytical is faster");
    println!("  A.Best/N.Best ~ 1.0 means similar optimization quality");
    println!();

    // Detailed Rastrigin test with more particles
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("DETAILED TEST: Rastrigin 8D with increasing particles");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let particle_counts = [500, 1000, 2000, 5000, 10000];

    println!("{:>10} {:>14} {:>14} {:>12} {:>14} {:>14}",
             "Particles", "A.µs/step", "N.µs/step", "Speedup", "A.Best", "N.Best");
    println!("{}", "-".repeat(82));

    for &n in &particle_counts {
        let analytical = benchmark_expr(rastrigin(), true, n, 8, 30, 300);
        let numerical = benchmark_expr(rastrigin(), false, n, 8, 30, 300);

        let speedup = numerical.us_per_step / analytical.us_per_step;

        println!("{:>10} {:>12.1}µs {:>12.1}µs {:>11.2}x {:>14.6} {:>14.6}",
                 n,
                 analytical.us_per_step,
                 numerical.us_per_step,
                 speedup,
                 analytical.best_energy,
                 numerical.best_energy);
    }

    println!();
    println!("Summary:");
    println!("  - Analytical gradients eliminate finite-difference overhead");
    println!("  - Speedup varies based on function complexity");
    println!("  - Accuracy is identical or better (no epsilon approximation error)");
}

struct BenchResult {
    us_per_step: f64,
    best_energy: f32,
}

fn benchmark_expr(
    expr: Expr,
    analytical: bool,
    particle_count: usize,
    dim: usize,
    warmup_steps: usize,
    bench_steps: usize,
) -> BenchResult {
    // Create system with the specified gradient type
    let mut system = ThermodynamicSystem::with_expr_options(
        particle_count,
        dim,
        2.0,
        expr,
        analytical,
    );
    system.set_repulsion_samples(32);

    // Warmup
    for _ in 0..warmup_steps {
        system.step();
    }

    // Benchmark with simulated annealing
    let start = Instant::now();
    for step in 0..bench_steps {
        let progress = step as f32 / bench_steps as f32;
        let temp = 2.0 * (0.001_f32 / 2.0).powf(progress);
        system.set_temperature(temp);
        system.step();
    }
    let elapsed = start.elapsed();

    let particles = system.read_particles();
    let best_energy = particles
        .iter()
        .filter(|p| !p.energy.is_nan())
        .map(|p| p.energy)
        .fold(f32::MAX, f32::min);

    BenchResult {
        us_per_step: elapsed.as_micros() as f64 / bench_steps as f64,
        best_energy,
    }
}
