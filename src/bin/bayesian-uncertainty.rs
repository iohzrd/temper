//! Bayesian Uncertainty Quantification Demo
//!
//! Demonstrates what standard gradient descent CAN'T do:
//! Instead of finding a single optimal weight configuration,
//! we sample from the posterior p(weights|data) ∝ exp(-loss/T)
//! to get uncertainty estimates on predictions.
//!
//! At temperature T ~ 0.1-1.0, particles spread across the posterior,
//! and the SVGD repulsion prevents mode collapse.
//!
//! Run with: cargo run --release --features gpu --bin bayesian-uncertainty

use temper::{LossFunction, ThermodynamicSystem};

const PARTICLE_COUNT: usize = 200;
const DIM: usize = 9; // 2->2->1 MLP: 2*2 + 2 + 2*1 + 1 = 9 params

fn main() {
    println!("Bayesian Uncertainty Quantification");
    println!("====================================\n");
    println!("Task: Train MLP on XOR, then sample posterior for uncertainty.\n");

    // Using XOR dataset: 4 points, binary classification
    // XOR: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
    println!("Dataset: XOR (4 points)");
    println!("Network: 2 -> 2 -> 1 (9 parameters)\n");

    // Phase 1: Optimization - find good weights
    println!("Phase 1: Optimization (T → 0)");
    println!("-----------------------------");

    let mut system =
        ThermodynamicSystem::with_loss_function(PARTICLE_COUNT, DIM, 2.0, LossFunction::MlpXor);

    // Anneal to find good weights
    let anneal_steps = 2000;
    for step in 0..anneal_steps {
        let progress = step as f32 / anneal_steps as f32;
        let temp = 2.0 * (0.001_f32 / 2.0).powf(progress);
        system.set_temperature(temp);
        system.step();

        if step % 500 == 0 {
            let particles = system.read_particles();
            let min_e = particles
                .iter()
                .filter(|p| !p.energy.is_nan())
                .map(|p| p.energy)
                .fold(f32::MAX, f32::min);
            println!("  Step {:4}: T={:.4}, min_loss={:.4}", step, temp, min_e);
        }
    }

    let particles = system.read_particles();
    let best_idx = particles
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.energy.is_nan())
        .min_by(|(_, a), (_, b)| a.energy.partial_cmp(&b.energy).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    println!(
        "\nBest particle found with loss: {:.6}",
        particles[best_idx].energy
    );

    // Phase 2: Posterior Sampling - spread particles across posterior
    println!("\nPhase 2: Posterior Sampling (T = 0.1)");
    println!("-------------------------------------");
    println!("Now we sample from p(weights|data) instead of optimizing.\n");

    // Reset to sampling temperature
    system.set_temperature(0.1);

    // Burn-in: let particles spread across posterior
    let burnin_steps = 1000;
    for step in 0..burnin_steps {
        system.step();

        if step % 250 == 0 {
            let particles = system.read_particles();
            let energies: Vec<f32> = particles
                .iter()
                .filter(|p| !p.energy.is_nan())
                .map(|p| p.energy)
                .collect();
            let mean_e: f32 = energies.iter().sum::<f32>() / energies.len() as f32;
            let var_e: f32 =
                energies.iter().map(|e| (e - mean_e).powi(2)).sum::<f32>() / energies.len() as f32;
            println!(
                "  Burn-in {:4}: mean_loss={:.4}, std_loss={:.4}",
                step,
                mean_e,
                var_e.sqrt()
            );
        }
    }

    // Collect posterior samples
    println!("\nCollecting posterior samples...");
    let sample_interval = 50;
    let n_samples = 20;
    let mut weight_samples: Vec<[f32; DIM]> = Vec::new();

    for i in 0..n_samples {
        for _ in 0..sample_interval {
            system.step();
        }
        let particles = system.read_particles();
        // Take top-k particles by energy as samples
        let mut sorted: Vec<_> = particles.iter().filter(|p| !p.energy.is_nan()).collect();
        sorted.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

        for p in sorted.iter().take(10) {
            let mut weights = [0.0f32; DIM];
            for d in 0..DIM {
                weights[d] = p.pos[d].to_f32();
            }
            weight_samples.push(weights);
        }

        if i % 5 == 0 {
            println!(
                "  Collected batch {} ({} total samples)",
                i + 1,
                weight_samples.len()
            );
        }
    }

    println!("\nTotal posterior samples: {}", weight_samples.len());

    // Phase 3: Predictive Uncertainty on XOR
    println!("\nPhase 3: Predictive Uncertainty on XOR");
    println!("---------------------------------------");
    println!("Testing predictions with uncertainty from posterior samples.\n");

    // XOR test points: training data + interpolation + extrapolation
    let test_points = [
        // Training data (should be confident)
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        // Interpolation (moderate uncertainty)
        (0.5, 0.5, 0.5), // center point
        (0.25, 0.75, 0.75),
        // Extrapolation (should show high uncertainty)
        (2.0, 0.0, -1.0), // unknown - no true label
        (-1.0, 1.0, -1.0),
        (1.5, 1.5, -1.0),
    ];

    println!(
        "{:>6} {:>6} {:>8} {:>10} {:>10} {:>10}",
        "x1", "x2", "y_true", "y_mean", "y_std", "region"
    );
    println!("{}", "-".repeat(62));

    for (x1, x2, y_true) in test_points {
        // Forward pass through network for each weight sample
        let predictions: Vec<f32> = weight_samples
            .iter()
            .map(|w| forward_pass_2d(x1, x2, w))
            .collect();

        let y_mean: f32 = predictions.iter().sum::<f32>() / predictions.len() as f32;
        let y_var: f32 = predictions
            .iter()
            .map(|&y| (y - y_mean).powi(2))
            .sum::<f32>()
            / predictions.len() as f32;
        let y_std = y_var.sqrt();

        let region = if y_true >= 0.0 {
            if (x1 == 0.0 || x1 == 1.0) && (x2 == 0.0 || x2 == 1.0) {
                "training"
            } else {
                "interp"
            }
        } else {
            "extrap"
        };

        let y_true_str = if y_true >= 0.0 {
            format!("{:.1}", y_true)
        } else {
            "?".to_string()
        };
        println!(
            "{:6.2} {:6.2} {:>8} {:10.4} {:10.4} {:>10}",
            x1, x2, y_true_str, y_mean, y_std, region
        );
    }

    println!("\nKey observations:");
    println!("  - Training points: low uncertainty (we've seen them)");
    println!("  - Interpolation: moderate uncertainty");
    println!("  - Extrapolation: HIGH uncertainty (network doesn't know!)");
    println!();
    println!("Key insight:");
    println!("  - Gradient descent: ONE weight vector → no uncertainty estimate");
    println!("  - Thermodynamic sampling: DISTRIBUTION over weights → uncertainty!");
    println!("  - The uncertainty tells you WHERE the model is confident");
}

/// 2->2->1 MLP forward pass matching the shader's mlp_xor architecture
fn forward_pass_2d(x1: f32, x2: f32, w: &[f32; 9]) -> f32 {
    // Layer 1: 2->2
    // Weights: w[0..4] (2x2 matrix), Biases: w[4..6]
    let h1_pre = w[0] * x1 + w[1] * x2 + w[4];
    let h2_pre = w[2] * x1 + w[3] * x2 + w[5];
    let h1 = h1_pre.tanh();
    let h2 = h2_pre.tanh();

    // Layer 2: 2->1
    // Weights: w[6..8], Bias: w[8]
    let out = w[6] * h1 + w[7] * h2 + w[8];

    // Sigmoid for probability
    1.0 / (1.0 + (-out).exp())
}
