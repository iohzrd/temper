//! Custom Expression Loss Function Demo
//!
//! Demonstrates the Expression DSL for defining custom loss functions
//! that compile to GPU-accelerated WGSL.
//!
//! Run with: cargo run --release --features gpu --bin custom-expr-demo

use temper::ThermodynamicSystem;
use temper::expr::*;

fn main() {
    println!("Custom Expression Loss Function Demo");
    println!("=====================================\n");

    // Define the Griewank function using the expression DSL:
    // f(x) = 1 + sum(x_i^2 / 4000) - prod(cos(x_i / sqrt(i+1)))
    // Global minimum: f(0, 0, ..., 0) = 0
    let griewank = const_(1.0) + sum_dims(|x, _| x.powi(2) / 4000.0)
        - prod_dims(|x, i| cos(x / sqrt(i + 1.0)));

    println!("Testing Griewank function (custom expression)");
    println!("Global minimum at origin: f(0,0,...,0) = 0\n");

    // Print the generated WGSL code
    println!("Generated WGSL:");
    println!("---------------");
    let wgsl = griewank.to_wgsl();
    for line in wgsl.lines().take(10) {
        println!("  {}", line);
    }
    println!("  ...\n");

    // Create system with custom expression
    let dim = 4;
    let particle_count = 500;
    let mut system = ThermodynamicSystem::with_expr(particle_count, dim, 2.0, griewank);
    system.set_repulsion_samples(64);

    println!(
        "Running optimization with {} particles in {}D...",
        particle_count, dim
    );
    println!();

    // Simulated annealing
    let steps = 3000;
    let t_start: f32 = 2.0;
    let t_end: f32 = 0.001;

    for step in 0..steps {
        let progress = step as f32 / steps as f32;
        let temp = t_start * (t_end / t_start).powf(progress);
        system.set_temperature(temp);

        // Adaptive time step
        let dt = if temp > 0.1 {
            0.01
        } else if temp > 0.01 {
            0.005
        } else {
            0.002
        };
        system.set_dt(dt);

        system.step();

        // Report progress
        if step % 500 == 0 || step == steps - 1 {
            let particles = system.read_particles();
            let min_energy = particles
                .iter()
                .filter(|p| !p.energy.is_nan())
                .map(|p| p.energy)
                .fold(f32::MAX, f32::min);

            let mean_energy = particles
                .iter()
                .filter(|p| !p.energy.is_nan())
                .map(|p| p.energy)
                .sum::<f32>()
                / particles.len() as f32;

            println!(
                "Step {:4}: T={:.4}  min={:.4}  mean={:.4}",
                step, temp, min_energy, mean_energy
            );
        }
    }

    // Final results
    println!();
    let particles = system.read_particles();
    let best = particles
        .iter()
        .filter(|p| !p.energy.is_nan())
        .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
        .unwrap();

    println!("Final Results");
    println!("-------------");
    println!("Best energy: {:.6}", best.energy);
    print!("Best position: [");
    for i in 0..dim {
        if i > 0 {
            print!(", ");
        }
        print!("{:.4}", best.pos[i].to_f32());
    }
    println!("]");

    let distance_from_origin: f32 = (0..dim)
        .map(|i| best.pos[i].to_f32().powi(2))
        .sum::<f32>()
        .sqrt();
    println!("Distance from origin: {:.6}", distance_from_origin);

    // Test with Levy function too
    println!("\n\nTesting Levy function");
    println!("---------------------");
    println!("Global minimum at (1, 1, ..., 1) = 0\n");

    let levy_expr = levy();
    let mut system2 = ThermodynamicSystem::with_expr(particle_count, dim, 2.0, levy_expr);
    system2.set_repulsion_samples(64);

    for step in 0..steps {
        let progress = step as f32 / steps as f32;
        let temp = t_start * (t_end / t_start).powf(progress);
        system2.set_temperature(temp);
        let dt = if temp > 0.1 {
            0.01
        } else if temp > 0.01 {
            0.005
        } else {
            0.002
        };
        system2.set_dt(dt);
        system2.step();
    }

    let particles2 = system2.read_particles();
    let best2 = particles2
        .iter()
        .filter(|p| !p.energy.is_nan())
        .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
        .unwrap();

    println!("Best energy: {:.6}", best2.energy);
    print!("Best position: [");
    for i in 0..dim {
        if i > 0 {
            print!(", ");
        }
        print!("{:.4}", best2.pos[i].to_f32());
    }
    println!("]");

    let distance_from_ones: f32 = (0..dim)
        .map(|i| (best2.pos[i].to_f32() - 1.0).powi(2))
        .sum::<f32>()
        .sqrt();
    println!("Distance from (1,1,1,1): {:.6}", distance_from_ones);

    println!("\nDemo complete!");
}
