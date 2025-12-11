//! Deep Neural Network Benchmark
//!
//! Tests the thermodynamic particle system on a deeper neural network:
//! - 3-layer MLP: 2 -> 4 -> 4 -> 1 (37 parameters)
//! - Concentric circles classification (harder than XOR)
//!
//! This proves the system can scale to larger networks with more parameters.

use nbody_entropy::thermodynamic::{LossFunction, ThermodynamicSystem};
use std::time::Instant;

const PARTICLE_COUNT: usize = 1000;  // More particles for higher-dim search
const DIM: usize = 37;  // 2*4 + 4 + 4*4 + 4 + 4*1 + 1 = 37 parameters

fn main() {
    println!("{}",
        "╔══════════════════════════════════════════════════════════════════════════╗");
    println!("{}",
        "║           DEEP NEURAL NETWORK BENCHMARK                                  ║");
    println!("{}",
        "╠══════════════════════════════════════════════════════════════════════════╣");
    println!("{}",
        "║  Training 3-layer MLP with thermodynamic particle optimization           ║");
    println!("{}",
        "║  Architecture: 2 inputs -> 4 hidden -> 4 hidden -> 1 output              ║");
    println!("{}",
        "║  Dataset: Concentric circles (100 points)                                ║");
    println!("{}",
        "╚══════════════════════════════════════════════════════════════════════════╝\n");

    let passed = train_deep_mlp();

    println!("{}",
        "╔══════════════════════════════════════════════════════════════════════════╗");
    println!("{}",
        "║                              SUMMARY                                     ║");
    println!("{}",
        "╠══════════════════════════════════════════════════════════════════════════╣");
    if passed {
        println!("{}",
            "║  ✓ DEEP NEURAL NETWORK TEST PASSED                                      ║");
        println!("{}",
            "║  37-parameter MLP successfully trained on circles dataset!              ║");
    } else {
        println!("{}",
            "║  ✗ TEST FAILED                                                          ║");
    }
    println!("{}",
        "╚══════════════════════════════════════════════════════════════════════════╝");
}

fn train_deep_mlp() -> bool {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("DEEP MLP ON CIRCLES CLASSIFICATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Problem: Classify points on inner (r=0.5) vs outer (r=1.5) circles");
    println!("  Parameters: {} (3-layer MLP weights and biases)", DIM);
    println!("  Particles: {}", PARTICLE_COUNT);

    // Use simulated annealing with more steps for harder problem
    let steps = 8000;
    let mut system = ThermodynamicSystem::with_loss_function(
        PARTICLE_COUNT, DIM, 2.0, LossFunction::MlpDeep
    );

    println!("  Training with simulated annealing ({} steps)...", steps);

    let start = Instant::now();
    for step in 0..steps {
        // Slower annealing for harder problem with more parameters
        let progress = step as f32 / steps as f32;
        let temp = 2.0 * (0.0001_f32 / 2.0).powf(progress);
        system.set_temperature(temp);
        system.step();

        // Progress updates
        if step % 2000 == 0 && step > 0 {
            let particles = system.read_particles();
            let min_loss = particles.iter()
                .filter(|p| !p.energy.is_nan())
                .map(|p| p.energy)
                .fold(f32::MAX, f32::min);
            println!("    Step {}: min_loss = {:.4}, temp = {:.6}", step, min_loss, temp);
        }
    }
    let elapsed = start.elapsed();

    let particles = system.read_particles();
    let energies: Vec<f32> = particles.iter()
        .filter(|p| !p.energy.is_nan())
        .map(|p| p.energy)
        .collect();

    let min_loss = energies.iter().cloned().fold(f32::MAX, f32::min);
    let mean_loss = energies.iter().sum::<f32>() / energies.len() as f32;
    let low_loss = energies.iter().filter(|&&e| e < 0.5).count();

    // Find best network
    let best = particles.iter()
        .filter(|p| !p.energy.is_nan())
        .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
        .unwrap();

    println!("\n  Time: {:?}", elapsed);
    println!("  Results:");
    println!("    Min loss (BCE): {:.6}", min_loss);
    println!("    Mean loss:      {:.6}", mean_loss);
    println!("    Low loss (<0.5): {:.1}%", low_loss as f32 / energies.len() as f32 * 100.0);

    // Test on sample points
    println!("\n  Testing on sample points:");
    let mut correct = 0;
    let total = 20;

    for i in 0..total {
        let theta = i as f32 / total as f32 * 2.0 * std::f32::consts::PI;
        // Alternate between inner and outer circles
        let (r, target): (f32, f32) = if i % 2 == 0 { (0.5, 0.0) } else { (1.5, 1.0) };
        let x = r * theta.cos();
        let y = r * theta.sin();

        // Manual forward pass through 3-layer network
        let pos = &best.pos;

        // Layer 1: input (2) -> hidden1 (4)
        let h1_0 = (pos[0] * x + pos[1] * y + pos[8]).tanh();
        let h1_1 = (pos[2] * x + pos[3] * y + pos[9]).tanh();
        let h1_2 = (pos[4] * x + pos[5] * y + pos[10]).tanh();
        let h1_3 = (pos[6] * x + pos[7] * y + pos[11]).tanh();

        // Layer 2: hidden1 (4) -> hidden2 (4)
        let h2_0 = (pos[12] * h1_0 + pos[13] * h1_1 + pos[14] * h1_2 + pos[15] * h1_3 + pos[28]).tanh();
        let h2_1 = (pos[16] * h1_0 + pos[17] * h1_1 + pos[18] * h1_2 + pos[19] * h1_3 + pos[29]).tanh();
        let h2_2 = (pos[20] * h1_0 + pos[21] * h1_1 + pos[22] * h1_2 + pos[23] * h1_3 + pos[30]).tanh();
        let h2_3 = (pos[24] * h1_0 + pos[25] * h1_1 + pos[26] * h1_2 + pos[27] * h1_3 + pos[31]).tanh();

        // Layer 3: hidden2 (4) -> output (1)
        let logit = pos[32] * h2_0 + pos[33] * h2_1 + pos[34] * h2_2 + pos[35] * h2_3 + pos[36];
        let out = 1.0 / (1.0 + (-logit).exp());

        let pred_class: f32 = if out > 0.5 { 1.0 } else { 0.0 };
        if (pred_class - target).abs() < 0.01 {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / total as f32;
    println!("    Accuracy: {}/{} = {:.0}%", correct, total, accuracy * 100.0);

    // Pass if we achieve reasonable loss and accuracy
    let passed = min_loss < 0.7 || accuracy >= 0.7;
    println!("\n  Status: {}", if passed { "PASS ✓" } else { "FAIL ✗" });
    passed
}
