//! Large Network Benchmark (200+ Parameters)
//!
//! Tests the thermodynamic optimizer on a substantial neural network:
//! - Architecture: 16 inputs → 8 hidden → 8 hidden → 4 outputs
//! - Parameters: 244 total (weights + biases)
//! - Task: 4-class classification on synthetic data
//!
//! This demonstrates the system's ability to optimize in high-dimensional
//! parameter spaces. MAX_DIMENSIONS is now 1024, supporting much larger networks.

use std::time::Instant;
use temper::ThermodynamicSystem;
use temper::expr::*;

// Network architecture:
// Layer 1: 16 inputs → 8 hidden (tanh): 16*8 + 8 = 136 params
// Layer 2: 8 hidden → 8 hidden (tanh): 8*8 + 8 = 72 params
// Layer 3: 8 hidden → 4 outputs (softmax): 8*4 + 4 = 36 params
// Total: 136 + 72 + 36 = 244 parameters

const INPUT_DIM: usize = 16;
const HIDDEN1: usize = 8;
const HIDDEN2: usize = 8;
const OUTPUT_DIM: usize = 4;

// Parameter layout:
// W1: 0..128 (16x8)
// b1: 128..136
// W2: 136..200 (8x8)
// b2: 200..208
// W3: 208..240 (8x4)
// b3: 240..244

const DIM: usize = 244;

fn main() {
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║                LARGE NETWORK BENCHMARK (244 Parameters)                  ║"
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "{}",
        "║  Architecture: 16 → 8 → 8 → 4 (tanh/tanh/softmax)                        ║"
    );
    println!(
        "{}",
        "║  Task: 4-class classification on synthetic clustered data                ║"
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝\n"
    );

    println!("Network Configuration:");
    println!("  Input dim:  {}", INPUT_DIM);
    println!("  Hidden 1:   {} neurons", HIDDEN1);
    println!("  Hidden 2:   {} neurons", HIDDEN2);
    println!("  Output:     {} classes", OUTPUT_DIM);
    println!("  Total params: {}", DIM);
    println!();

    // Create synthetic dataset
    println!("Creating synthetic dataset...");
    let dataset = create_synthetic_dataset();
    println!(
        "  {} samples, {} features, {} classes",
        dataset.len(),
        INPUT_DIM,
        OUTPUT_DIM
    );
    println!();

    // Build the loss function using expression DSL
    // This creates WGSL code for the neural network forward pass + cross-entropy
    println!("Building custom neural network loss function...");
    let loss_fn = build_nn_loss(&dataset);
    println!("  Generated WGSL shader for {}D optimization", DIM);
    println!();

    // Run optimization
    let particle_count = 1000;
    let steps = 3000;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("THERMODYNAMIC OPTIMIZATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Particles: {}", particle_count);
    println!("  Steps: {}", steps);
    println!("  Annealing: T = 2.0 → 0.0001");
    println!();

    let mut system = ThermodynamicSystem::with_expr(particle_count, DIM, 2.0, loss_fn);

    let start = Instant::now();
    let mut best_loss = f32::MAX;

    for step in 0..steps {
        // Simulated annealing schedule
        let progress = step as f32 / steps as f32;
        let temp = 2.0 * (0.0001_f32 / 2.0).powf(progress);
        system.set_temperature(temp);
        system.step();

        // Report progress
        if (step + 1) % 500 == 0 || step == 0 {
            let particles = system.read_particles();
            let min_loss = particles
                .iter()
                .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
                .map(|p| p.energy)
                .fold(f32::MAX, f32::min);

            if min_loss < best_loss {
                best_loss = min_loss;
            }

            let diversity = system.diversity_metrics();

            println!(
                "  Step {:5}: loss={:.4}, best={:.4}, T={:.6}, ESS={:.0}, modes={}",
                step + 1,
                min_loss,
                best_loss,
                temp,
                diversity.effective_sample_size,
                diversity.estimated_modes
            );
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!("  Total time: {:?}", elapsed);
    println!("  Steps/sec: {:.0}", steps as f64 / elapsed.as_secs_f64());
    println!();

    // Evaluate best network
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("EVALUATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let particles = system.read_particles();
    let best = particles
        .iter()
        .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
        .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
        .unwrap();

    println!("  Best loss: {:.6}", best.energy);

    // Extract weights and evaluate accuracy
    let weights: Vec<f32> = (0..DIM).map(|i| best.pos[i].to_f32()).collect();
    let accuracy = evaluate_accuracy(&weights, &dataset);
    println!("  Train accuracy: {:.1}%", accuracy * 100.0);

    // Final diversity metrics
    let diversity = system.diversity_metrics();
    println!();
    println!("  Final Diversity Metrics:");
    println!(
        "    Mean pairwise distance: {:.4}",
        diversity.mean_pairwise_distance
    );
    println!(
        "    Effective sample size: {:.0}",
        diversity.effective_sample_size
    );
    println!("    Estimated modes: {}", diversity.estimated_modes);
    println!("    Coverage: {:.4}", diversity.coverage);

    // Summary
    println!();
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    if accuracy >= 0.7 {
        println!(
            "{}",
            "║  ✓ LARGE NETWORK BENCHMARK PASSED                                        ║"
        );
        println!(
            "{}",
            "║  Successfully trained 244-parameter network with thermodynamic optimizer ║"
        );
    } else {
        println!(
            "{}",
            "║  ✗ BENCHMARK DID NOT REACH TARGET ACCURACY                               ║"
        );
        println!(
            "{}",
            "║  Consider increasing particles or steps                                  ║"
        );
    }
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝"
    );
}

/// Synthetic sample: features and one-hot encoded label
struct Sample {
    features: [f32; INPUT_DIM],
    label: usize,
}

/// Create synthetic clustered data for 4-class classification
fn create_synthetic_dataset() -> Vec<Sample> {
    let mut samples = Vec::new();
    let samples_per_class = 50;

    // Class centers in 16D space
    // Each class has a distinct pattern in different dimensions
    for class in 0..OUTPUT_DIM {
        for i in 0..samples_per_class {
            let mut features = [0.0f32; INPUT_DIM];

            // Base pattern for each class
            for d in 0..INPUT_DIM {
                // Each class emphasizes different dimensions
                let base = if (d / 4) == class { 0.8 } else { 0.2 };

                // Deterministic noise based on sample index
                let seed = (class * 1000 + i * 100 + d) as u32;
                let noise = ((seed ^ (seed >> 7) ^ (seed << 3)) % 1000) as f32 / 5000.0 - 0.1;

                features[d] = (base + noise).clamp(0.0, 1.0);
            }

            samples.push(Sample {
                features,
                label: class,
            });
        }
    }

    samples
}

/// Build WGSL loss function for the neural network
fn build_nn_loss(dataset: &[Sample]) -> temper::expr::Expr {
    // Generate WGSL code for the neural network loss
    // This is more efficient than using the expression DSL for complex networks

    let mut loss_code = String::new();
    let mut grad_code = String::new();

    // Generate loss function
    loss_code.push_str("fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {\n");
    loss_code.push_str("    var total_loss = 0.0;\n\n");

    // Embed dataset as constants (use 40 samples for faster shader compilation)
    let n_samples = dataset.len().min(40);

    for (idx, sample) in dataset.iter().take(n_samples).enumerate() {
        loss_code.push_str(&format!("    // Sample {}\n", idx));
        loss_code.push_str("    {\n");

        // Input features
        for (d, &f) in sample.features.iter().enumerate() {
            loss_code.push_str(&format!("        let x{} = {:.4};\n", d, f));
        }

        // Layer 1: 16 → 8 (tanh)
        for h in 0..HIDDEN1 {
            let mut expr = String::new();
            for i in 0..INPUT_DIM {
                if i > 0 {
                    expr.push_str(" + ");
                }
                expr.push_str(&format!("pos[{}] * x{}", h * INPUT_DIM + i, i));
            }
            expr.push_str(&format!(" + pos[{}]", 128 + h)); // bias
            loss_code.push_str(&format!("        let h1_{} = tanh({});\n", h, expr));
        }

        // Layer 2: 8 → 8 (tanh)
        for h in 0..HIDDEN2 {
            let mut expr = String::new();
            for i in 0..HIDDEN1 {
                if i > 0 {
                    expr.push_str(" + ");
                }
                expr.push_str(&format!("pos[{}] * h1_{}", 136 + h * HIDDEN1 + i, i));
            }
            expr.push_str(&format!(" + pos[{}]", 200 + h)); // bias
            loss_code.push_str(&format!("        let h2_{} = tanh({});\n", h, expr));
        }

        // Layer 3: 8 → 4 (logits for softmax)
        for o in 0..OUTPUT_DIM {
            let mut expr = String::new();
            for i in 0..HIDDEN2 {
                if i > 0 {
                    expr.push_str(" + ");
                }
                expr.push_str(&format!("pos[{}] * h2_{}", 208 + o * HIDDEN2 + i, i));
            }
            expr.push_str(&format!(" + pos[{}]", 240 + o)); // bias
            loss_code.push_str(&format!("        let z{} = {};\n", o, expr));
        }

        // Stable softmax + cross-entropy
        loss_code.push_str("        let max_z = max(max(max(z0, z1), z2), z3);\n");
        for o in 0..OUTPUT_DIM {
            loss_code.push_str(&format!("        let e{} = exp(z{} - max_z);\n", o, o));
        }
        loss_code.push_str("        let sum_e = e0 + e1 + e2 + e3;\n");

        // Cross-entropy for correct class
        loss_code.push_str(&format!(
            "        total_loss = total_loss - log(e{} / sum_e + 0.0001);\n",
            sample.label
        ));

        loss_code.push_str("    }\n\n");
    }

    loss_code.push_str(&format!("    return total_loss / {}.0;\n", n_samples));
    loss_code.push_str("}\n");

    // Numerical gradient (simpler than analytical for this size)
    grad_code.push_str("fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {\n");
    grad_code.push_str(&format!("    if d_idx >= {}u {{\n", DIM));
    grad_code.push_str("        return 0.0;\n");
    grad_code.push_str("    }\n");
    grad_code.push_str("    let eps = 0.001;\n");
    grad_code.push_str("    var pos_plus = pos;\n");
    grad_code.push_str("    var pos_minus = pos;\n");
    grad_code.push_str("    pos_plus[d_idx] = pos[d_idx] + eps;\n");
    grad_code.push_str("    pos_minus[d_idx] = pos[d_idx] - eps;\n");
    grad_code.push_str(
        "    return (custom_loss(pos_plus, dim) - custom_loss(pos_minus, dim)) / (2.0 * eps);\n",
    );
    grad_code.push_str("}\n");

    custom_wgsl(&loss_code, &grad_code)
}

/// Forward pass through the network (CPU version for evaluation)
fn forward(weights: &[f32], features: &[f32; INPUT_DIM]) -> [f32; OUTPUT_DIM] {
    // Layer 1: 16 → 8
    let mut h1 = [0.0f32; HIDDEN1];
    for h in 0..HIDDEN1 {
        let mut sum = weights[128 + h]; // bias
        for i in 0..INPUT_DIM {
            sum += weights[h * INPUT_DIM + i] * features[i];
        }
        h1[h] = sum.tanh();
    }

    // Layer 2: 8 → 8
    let mut h2 = [0.0f32; HIDDEN2];
    for h in 0..HIDDEN2 {
        let mut sum = weights[200 + h]; // bias
        for i in 0..HIDDEN1 {
            sum += weights[136 + h * HIDDEN1 + i] * h1[i];
        }
        h2[h] = sum.tanh();
    }

    // Layer 3: 8 → 4
    let mut logits = [0.0f32; OUTPUT_DIM];
    for o in 0..OUTPUT_DIM {
        let mut sum = weights[240 + o]; // bias
        for i in 0..HIDDEN2 {
            sum += weights[208 + o * HIDDEN2 + i] * h2[i];
        }
        logits[o] = sum;
    }

    // Softmax
    let max_logit = logits.iter().cloned().fold(f32::MIN, f32::max);
    let mut probs = [0.0f32; OUTPUT_DIM];
    let mut sum_exp = 0.0f32;
    for o in 0..OUTPUT_DIM {
        probs[o] = (logits[o] - max_logit).exp();
        sum_exp += probs[o];
    }
    for o in 0..OUTPUT_DIM {
        probs[o] /= sum_exp;
    }

    probs
}

/// Evaluate classification accuracy
fn evaluate_accuracy(weights: &[f32], dataset: &[Sample]) -> f32 {
    let mut correct = 0;
    for sample in dataset {
        let probs = forward(weights, &sample.features);
        let predicted = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        if predicted == sample.label {
            correct += 1;
        }
    }
    correct as f32 / dataset.len() as f32
}
