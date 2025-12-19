//! Multi-Modal Mode Discovery Demo
//!
//! Demonstrates finding ALL modes of a multi-modal distribution,
//! not just converging to one like gradient descent does.
//!
//! The SVGD repulsion term is crucial: it pushes particles apart,
//! preventing mode collapse and ensuring coverage of the full landscape.
//!
//! Run with: cargo run --release --features gpu --bin mode-discovery

use std::collections::HashMap;
use temper::ThermodynamicSystem;
use temper::expr::*;

fn main() {
    println!("Multi-Modal Mode Discovery");
    println!("==========================\n");

    // Create a 2D energy landscape with 4 known minima
    // E(x,y) = (x² - 4)² + (y² - 4)² - creates minima at (±2, ±2)
    println!("Energy landscape: E(x,y) = (x² - 4)² + (y² - 4)²");
    println!("Known minima at: (-2,-2), (-2,+2), (+2,-2), (+2,+2)\n");

    let four_wells = sum_dims(|x, _| (x.powi(2) - const_(4.0)).powi(2));

    // Method 1: Standard gradient descent (simulated by T→0)
    println!("Method 1: Gradient Descent (T → 0, no repulsion)");
    println!("-------------------------------------------------");

    let mut gd_system = ThermodynamicSystem::with_expr(100, 2, 0.001, four_wells.clone());

    // Run for convergence
    for _ in 0..2000 {
        gd_system.step();
    }

    let gd_particles = gd_system.read_particles();
    let gd_modes = find_modes(&gd_particles, 2);

    println!("Particles converged to {} mode(s):", gd_modes.len());
    for (mode, count) in &gd_modes {
        println!("  ({:+.2}, {:+.2}): {} particles", mode.0, mode.1, count);
    }
    println!();

    // Method 2: Thermodynamic sampling with SVGD
    println!("Method 2: Thermodynamic Sampling (T = 0.5, SVGD repulsion)");
    println!("-----------------------------------------------------------");

    let mut thermo_system = ThermodynamicSystem::with_expr(100, 2, 2.0, four_wells.clone());

    // Anneal but stay at moderate temperature
    for step in 0..3000 {
        let progress = step as f32 / 3000.0;
        let temp = 2.0 * (0.1_f32 / 2.0).powf(progress); // Anneal to T=0.1
        thermo_system.set_temperature(temp);
        thermo_system.step();
    }

    let thermo_particles = thermo_system.read_particles();
    let thermo_modes = find_modes(&thermo_particles, 2);

    println!("Particles spread across {} mode(s):", thermo_modes.len());
    for (mode, count) in &thermo_modes {
        println!("  ({:+.2}, {:+.2}): {} particles", mode.0, mode.1, count);
    }
    println!();

    // Method 3: Higher temperature for better exploration
    println!("Method 3: Higher Temperature (T = 1.0, SVGD repulsion)");
    println!("-------------------------------------------------------");

    let mut hot_system = ThermodynamicSystem::with_expr(200, 2, 3.0, four_wells.clone());

    // Gentle annealing
    for step in 0..4000 {
        let progress = step as f32 / 4000.0;
        let temp = 3.0 * (0.3_f32 / 3.0).powf(progress);
        hot_system.set_temperature(temp);
        hot_system.step();
    }

    let hot_particles = hot_system.read_particles();
    let hot_modes = find_modes(&hot_particles, 2);

    println!("Particles spread across {} mode(s):", hot_modes.len());
    for (mode, count) in &hot_modes {
        println!("  ({:+.2}, {:+.2}): {} particles", mode.0, mode.1, count);
    }
    println!();

    // Now demonstrate with a higher-dimensional problem
    println!("{}", "=".repeat(60));
    println!("\nHigher Dimensional Test: 4D with 16 modes");
    println!("------------------------------------------");
    println!("Energy: sum_i (x_i² - 1)²  → modes at all corners of {{-1,+1}}^4\n");

    let hypercube_wells = sum_dims(|x, _| (x.powi(2) - const_(1.0)).powi(2));

    let mut hd_system = ThermodynamicSystem::with_expr(500, 4, 2.0, hypercube_wells);

    for step in 0..5000 {
        let progress = step as f32 / 5000.0;
        let temp = 2.0 * (0.1_f32 / 2.0).powf(progress);
        hd_system.set_temperature(temp);
        hd_system.step();

        if step % 1000 == 0 {
            let particles = hd_system.read_particles();
            let modes_4d = find_modes_4d(&particles);
            println!(
                "Step {:4}: T={:.3}, modes discovered: {}/16",
                step, temp, modes_4d
            );
        }
    }

    let final_particles = hd_system.read_particles();
    let final_modes = find_modes_4d(&final_particles);

    println!(
        "\nFinal: Discovered {}/16 modes of the 4D hypercube",
        final_modes
    );
    println!("\nConclusion:");
    println!("  - Gradient descent: converges to ONE mode");
    println!("  - Thermodynamic + SVGD: discovers MULTIPLE modes");
    println!("  - Temperature controls exploration vs. exploitation");
    println!("  - Repulsion prevents mode collapse");
}

/// Find distinct modes in 2D by clustering particles
fn find_modes(particles: &[temper::ThermodynamicParticle], dim: usize) -> Vec<((f32, f32), usize)> {
    let threshold = 0.5; // Consider particles within 0.5 as same mode
    let mut mode_counts: HashMap<(i32, i32), usize> = HashMap::new();

    for p in particles
        .iter()
        .filter(|p| !p.energy.is_nan() && p.energy < 10.0)
    {
        // Round to grid
        let key = (
            (p.pos[0].to_f32() / threshold).round() as i32,
            if dim > 1 {
                (p.pos[1].to_f32() / threshold).round() as i32
            } else {
                0
            },
        );
        *mode_counts.entry(key).or_insert(0) += 1;
    }

    // Cluster nearby grid cells
    let mut modes: Vec<((f32, f32), usize)> = Vec::new();
    for ((gx, gy), count) in mode_counts {
        if count >= 3 {
            // At least 3 particles to count as a mode
            let x = gx as f32 * threshold;
            let y = gy as f32 * threshold;
            modes.push(((x, y), count));
        }
    }

    // Sort by count
    modes.sort_by(|a, b| b.1.cmp(&a.1));
    modes
}

/// Count how many of the 16 corners of {-1,+1}^4 have nearby particles
fn find_modes_4d(particles: &[temper::ThermodynamicParticle]) -> usize {
    let threshold = 0.5;
    let mut found = std::collections::HashSet::new();

    for p in particles
        .iter()
        .filter(|p| !p.energy.is_nan() && p.energy < 1.0)
    {
        // Check which corner this is near
        let signs: Vec<i32> = (0..4)
            .map(|i| if p.pos[i].to_f32() > 0.0 { 1 } else { -1 })
            .collect();

        // Check if actually near a corner
        let near_corner = (0..4).all(|i| (p.pos[i].to_f32().abs() - 1.0).abs() < threshold);

        if near_corner {
            found.insert((signs[0], signs[1], signs[2], signs[3]));
        }
    }

    found.len()
}
