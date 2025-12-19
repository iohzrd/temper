//! High-Dimension Test
//!
//! Quick test to verify the system works at 200+ dimensions
//! using built-in loss functions (Rastrigin, Ackley).

use std::time::Instant;
use temper::thermodynamic::{LossFunction, ThermodynamicSystem};

fn main() {
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║                    HIGH-DIMENSION TEST (200+ dims)                       ║"
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝\n"
    );

    // Test various high dimensions with built-in loss functions
    let test_configs = [
        (200, "Rastrigin", LossFunction::Rastrigin),
        (220, "Sphere", LossFunction::Sphere),
        (244, "Ackley", LossFunction::Ackley),
        (256, "Rastrigin", LossFunction::Rastrigin),
    ];

    for (dim, name, loss_fn) in &test_configs {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("TEST: {}D {} function", dim, name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let particle_count = 1000;
        let steps = 1000;

        let mut system =
            ThermodynamicSystem::with_loss_function(particle_count, *dim, 2.0, loss_fn.clone());

        // Warmup
        for _ in 0..10 {
            system.step();
        }

        let start = Instant::now();
        let mut best_loss = f32::MAX;

        for step in 0..steps {
            let progress = step as f32 / steps as f32;
            let temp = 2.0 * (0.001_f32 / 2.0).powf(progress);
            system.set_temperature(temp);
            system.step();

            if (step + 1) % 200 == 0 {
                let particles = system.read_particles();
                let min_loss = particles
                    .iter()
                    .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
                    .map(|p| p.energy)
                    .fold(f32::MAX, f32::min);

                if min_loss < best_loss {
                    best_loss = min_loss;
                }
                println!(
                    "  Step {:4}: loss={:.2}, best={:.2}, T={:.5}",
                    step + 1,
                    min_loss,
                    best_loss,
                    temp
                );
            }
        }

        let elapsed = start.elapsed();

        // Final results
        let particles = system.read_particles();
        let final_min = particles
            .iter()
            .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
            .map(|p| p.energy)
            .fold(f32::MAX, f32::min);

        let valid_count = particles
            .iter()
            .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
            .count();

        // For Rastrigin, global min is 0. For Sphere, global min is 0.
        // For Ackley, global min is 0.
        let expected_min: f32 = 0.0;
        let _relative_error = final_min / (expected_min.abs() + 1.0);

        println!();
        println!("  Results:");
        println!("    Time: {:?}", elapsed);
        println!("    Steps/sec: {:.0}", steps as f64 / elapsed.as_secs_f64());
        println!("    Valid particles: {}/{}", valid_count, particle_count);
        println!("    Final min loss: {:.4}", final_min);
        println!(
            "    Status: {}",
            if valid_count > particle_count * 9 / 10 {
                "PASS ✓"
            } else {
                "FAIL ✗"
            }
        );
        println!();
    }

    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║                    HIGH-DIMENSION TEST COMPLETE                          ║"
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝"
    );
}
