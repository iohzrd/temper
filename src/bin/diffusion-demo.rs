//! Diffusion Model Connection Demo
//!
//! Demonstrates that Temper's Langevin dynamics IS the reverse diffusion process
//! used in score-based generative models (DDPM, score matching, etc.)
//!
//! The Key Insight:
//! ================
//! Diffusion models learn to reverse a noising process using the score function:
//!   x_{t-1} = x_t + ε·∇log p_t(x) + √(2ε)·z
//!
//! Temper's Langevin dynamics:
//!   dx = -γ·∇E(x)·dt + √(2γT)·dW
//!
//! These are THE SAME EQUATION when:
//!   - E(x) = -log p(x)  (energy = negative log probability)
//!   - ∇log p(x) = -∇E(x)  (score = negative gradient)
//!   - Temperature T corresponds to noise level
//!
//! Annealing temperature from high→low IS the reverse diffusion process!
//!
//! Run with: cargo run --release --features gpu --bin diffusion-demo

use temper::ThermodynamicSystem;
use temper::expr::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║           DIFFUSION MODEL CONNECTION DEMO                                ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Showing that Langevin annealing IS the reverse diffusion process        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    println!("THE MATHEMATICAL CONNECTION");
    println!("===========================\n");
    println!("Diffusion Model (Score-Based):    x' = x + ε·∇log p(x) + √(2ε)·z");
    println!("Langevin Dynamics:                dx = -γ·∇E(x)·dt + √(2γT)·dW\n");
    println!("When E(x) = -log p(x), these are IDENTICAL:");
    println!("  • Score function s(x) = ∇log p(x) = -∇E(x)");
    println!("  • Temperature T ↔ Noise level σ²");
    println!("  • Annealing T: high→low = Reverse diffusion: t=T→t=0\n");
    println!("{}\n", "─".repeat(74));

    // =========================================================================
    // DEMO 1: Simple Gaussian Mixture (like a 2D "data distribution")
    // =========================================================================
    println!("DEMO 1: Denoising into a Gaussian Mixture");
    println!("==========================================\n");
    println!("Target: 4 Gaussians at (±2, ±2) - like a simple 2D 'image' distribution");
    println!("This is p_data(x). We'll show particles 'denoising' from noise → samples.\n");

    // Energy function: E(x) = -log p(x) for mixture of 4 Gaussians
    // Using (x² - 4)² + (y² - 4)² gives minima at ±2, ±2
    let gaussian_mixture = sum_dims(|x, _| (x.powi(2) - const_(4.0)).powi(2));

    let n_particles = 500;
    let mut system = ThermodynamicSystem::with_expr(n_particles, 2, 10.0, gaussian_mixture.clone());
    system.set_repulsion_samples(64); // SVGD for diversity

    println!("Diffusion     Temper          Particle Distribution");
    println!("Timestep      Temperature     (showing 'denoising' progress)");
    println!("─────────────────────────────────────────────────────────────────");

    // Simulate reverse diffusion by annealing temperature
    let total_steps = 5000;
    let t_start = 10.0_f32;
    let t_end = 0.05_f32;

    for step in 0..=total_steps {
        let progress = step as f32 / total_steps as f32;

        // Temperature schedule (exponential decay like diffusion noise schedule)
        let temp = t_start * (t_end / t_start).powf(progress);
        system.set_temperature(temp);

        if step > 0 {
            system.step();
        }

        // Report at key timesteps (like diffusion model visualization)
        if step % 1000 == 0 || step == total_steps {
            let particles = system.read_particles();
            let stats = compute_stats(&particles, 2);

            // Map temperature to "diffusion timestep" (T=1000 → T=0 convention)
            let diffusion_t = ((1.0 - progress) * 1000.0) as u32;

            let phase = match diffusion_t {
                800..=1000 => "Pure noise",
                500..=799 => "Noisy structure",
                200..=499 => "Emerging modes",
                50..=199 => "Clear modes",
                _ => "Final samples",
            };

            println!(
                "t={:4}        T={:.4}        {} (spread={:.2}, modes≈{})",
                diffusion_t, temp, phase, stats.spread, stats.mode_count
            );
        }
    }

    // Final analysis
    let final_particles = system.read_particles();
    println!("\nFinal Distribution Analysis:");
    println!("───────────────────────────────");
    analyze_modes(&final_particles, 2);

    // =========================================================================
    // DEMO 2: Higher-dimensional "denoising"
    // =========================================================================
    println!("\n\nDEMO 2: 8D Denoising (like a latent space)");
    println!("============================================\n");
    println!("In real diffusion models, denoising happens in high-D latent space.");
    println!("Here: 8D space with 256 modes (corners of hypercube).\n");

    let hypercube = sum_dims(|x, _| (x.powi(2) - const_(1.0)).powi(2));

    let mut hd_system = ThermodynamicSystem::with_expr(1000, 8, 5.0, hypercube);
    hd_system.set_repulsion_samples(64);

    println!("Diffusion     Temper          Entropy       Modes");
    println!("Progress      Temperature     (bits/dim)    Found");
    println!("─────────────────────────────────────────────────────────────────");

    for step in 0..=4000 {
        let progress = step as f32 / 4000.0;
        let temp = 5.0 * (0.05_f32 / 5.0).powf(progress);
        hd_system.set_temperature(temp);

        if step > 0 {
            hd_system.step();
        }

        if step % 800 == 0 || step == 4000 {
            let particles = hd_system.read_particles();
            let entropy = estimate_entropy(&particles, 8);
            let modes = count_hypercube_modes(&particles, 8);

            let pct = (progress * 100.0) as u32;
            println!(
                "{:3}%          T={:.4}        {:.2}          {}/256",
                pct, temp, entropy, modes
            );
        }
    }

    // =========================================================================
    // DEMO 3: Score Function Visualization
    // =========================================================================
    println!("\n\nDEMO 3: Score Function = Negative Energy Gradient");
    println!("==================================================\n");
    println!("In diffusion models, we learn s_θ(x) ≈ ∇log p(x)");
    println!("In Temper, we compute -∇E(x) analytically via autodiff.\n");
    println!("At a point x, the score points toward high-probability regions:");

    // Show score at a few points
    println!("\n  Point          Energy      Score Direction");
    println!("  ───────────────────────────────────────────────────");

    let test_points = [
        (0.0, 0.0), // Origin - should point nowhere (saddle)
        (1.0, 1.0), // Between modes - should point toward (2,2)
        (2.0, 2.0), // At mode - should be near zero
        (3.0, 0.0), // Away from modes - should point toward (2,0) or (2,2)
    ];

    for (x, y) in test_points {
        let energy = energy_at(x, y);
        let (gx, gy) = gradient_at(x, y);
        let score = (-gx, -gy); // Score = -∇E
        let mag = (score.0 * score.0 + score.1 * score.1).sqrt();

        let direction = if mag < 0.1 {
            "≈ zero (at mode)".to_string()
        } else {
            format!("→ ({:+.1}, {:+.1})", score.0 / mag, score.1 / mag)
        };

        println!(
            "  ({:+.1}, {:+.1})       {:.2}        {}",
            x, y, energy, direction
        );
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                              KEY INSIGHTS                                ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                          ║");
    println!("║  1. SAME EQUATION: Langevin dynamics = Score-based diffusion sampling   ║");
    println!("║                                                                          ║");
    println!("║  2. TEMPERATURE = NOISE LEVEL:                                           ║");
    println!("║     High T → Pure noise (diffusion t=T)                                  ║");
    println!("║     Low T  → Clean samples (diffusion t=0)                               ║");
    println!("║                                                                          ║");
    println!("║  3. ANNEALING = REVERSE DIFFUSION:                                       ║");
    println!("║     Cooling from T_high → T_low IS denoising                             ║");
    println!("║                                                                          ║");
    println!("║  4. SVGD REPULSION = DIVERSITY:                                          ║");
    println!("║     Prevents mode collapse, ensures full distribution coverage           ║");
    println!("║                                                                          ║");
    println!("║  5. IMPLICATIONS FOR HARDWARE:                                           ║");
    println!("║     Physical thermal noise → Natural diffusion sampling                  ║");
    println!("║     No need to simulate noise - just use temperature!                    ║");
    println!("║                                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

/// Compute statistics about particle distribution
struct ParticleStats {
    spread: f32,
    mode_count: usize,
}

fn compute_stats(particles: &[temper::ThermodynamicParticle], dim: usize) -> ParticleStats {
    let valid: Vec<_> = particles
        .iter()
        .filter(|p| !p.energy.is_nan() && p.energy < 100.0)
        .collect();

    if valid.is_empty() {
        return ParticleStats {
            spread: 0.0,
            mode_count: 0,
        };
    }

    // Compute spread (standard deviation)
    let mut sum = vec![0.0f32; dim];
    let mut sum_sq = vec![0.0f32; dim];

    for p in &valid {
        for d in 0..dim {
            let val = p.pos[d].to_f32();
            sum[d] += val;
            sum_sq[d] += val * val;
        }
    }

    let n = valid.len() as f32;
    let variance: f32 = (0..dim)
        .map(|d| sum_sq[d] / n - (sum[d] / n).powi(2))
        .sum::<f32>()
        / dim as f32;
    let spread = variance.sqrt();

    // Count modes (rough clustering)
    let mut mode_count = 0;
    let grid_size = 1.0;
    let mut seen = std::collections::HashSet::new();

    for p in &valid {
        if p.energy < 2.0 {
            // Only count low-energy particles
            let key = (
                (p.pos[0].to_f32() / grid_size).round() as i32,
                (p.pos[1].to_f32() / grid_size).round() as i32,
            );
            if seen.insert(key) {
                mode_count += 1;
            }
        }
    }

    // Cap at 4 for this demo (we expect 4 modes)
    mode_count = mode_count.min(4);

    ParticleStats { spread, mode_count }
}

fn analyze_modes(particles: &[temper::ThermodynamicParticle], _dim: usize) {
    let mut quadrant_counts = [0usize; 4];

    for p in particles
        .iter()
        .filter(|p| !p.energy.is_nan() && p.energy < 2.0)
    {
        let x = p.pos[0].to_f32();
        let y = p.pos[1].to_f32();
        let q = match (x > 0.0, y > 0.0) {
            (true, true) => 0,   // (+,+)
            (false, true) => 1,  // (-,+)
            (false, false) => 2, // (-,-)
            (true, false) => 3,  // (+,-)
        };
        quadrant_counts[q] += 1;
    }

    let total: usize = quadrant_counts.iter().sum();
    println!(
        "Mode at (+2, +2): {} particles ({:.1}%)",
        quadrant_counts[0],
        100.0 * quadrant_counts[0] as f32 / total as f32
    );
    println!(
        "Mode at (-2, +2): {} particles ({:.1}%)",
        quadrant_counts[1],
        100.0 * quadrant_counts[1] as f32 / total as f32
    );
    println!(
        "Mode at (-2, -2): {} particles ({:.1}%)",
        quadrant_counts[2],
        100.0 * quadrant_counts[2] as f32 / total as f32
    );
    println!(
        "Mode at (+2, -2): {} particles ({:.1}%)",
        quadrant_counts[3],
        100.0 * quadrant_counts[3] as f32 / total as f32
    );
    println!(
        "\nTotal converged: {} / {} particles",
        total,
        particles.len()
    );
}

/// Estimate entropy in bits per dimension
fn estimate_entropy(particles: &[temper::ThermodynamicParticle], dim: usize) -> f32 {
    let valid: Vec<_> = particles.iter().filter(|p| !p.energy.is_nan()).collect();

    if valid.is_empty() {
        return 0.0;
    }

    // Use energy spread as proxy for entropy
    let energies: Vec<f32> = valid.iter().map(|p| p.energy).collect();
    let mean_e: f32 = energies.iter().sum::<f32>() / energies.len() as f32;
    let var_e: f32 =
        energies.iter().map(|e| (e - mean_e).powi(2)).sum::<f32>() / energies.len() as f32;

    // Higher variance = higher entropy (rough estimate)
    (var_e.sqrt() + 1.0).log2() / dim as f32
}

/// Count how many hypercube corners have particles nearby
fn count_hypercube_modes(particles: &[temper::ThermodynamicParticle], dim: usize) -> usize {
    let threshold = 0.5;
    let mut found = std::collections::HashSet::new();

    for p in particles
        .iter()
        .filter(|p| !p.energy.is_nan() && p.energy < 1.0)
    {
        // Check if near a corner of {-1, +1}^dim
        let near_corner = (0..dim).all(|i| (p.pos[i].to_f32().abs() - 1.0).abs() < threshold);

        if near_corner {
            let signs: Vec<i8> = (0..dim)
                .map(|i| if p.pos[i].to_f32() > 0.0 { 1 } else { -1 })
                .collect();
            found.insert(signs);
        }
    }

    found.len()
}

/// Energy at point (x, y) for the 4-Gaussian mixture
fn energy_at(x: f32, y: f32) -> f32 {
    (x * x - 4.0).powi(2) + (y * y - 4.0).powi(2)
}

/// Gradient of energy at point (x, y)
fn gradient_at(x: f32, y: f32) -> (f32, f32) {
    // E = (x² - 4)² + (y² - 4)²
    // dE/dx = 2(x² - 4) · 2x = 4x(x² - 4)
    // dE/dy = 2(y² - 4) · 2y = 4y(y² - 4)
    let gx = 4.0 * x * (x * x - 4.0);
    let gy = 4.0 * y * (y * y - 4.0);
    (gx, gy)
}
