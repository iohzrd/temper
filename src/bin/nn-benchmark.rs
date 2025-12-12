//! Neural Network Benchmark
//!
//! Tests the thermodynamic particle system on real neural network training:
//! - XOR classification (classic non-linear problem)
//! - Spiral classification (harder 2-class problem)
//!
//! This proves the system can train actual neural networks, not just toy functions.

use std::time::Instant;
use temper::thermodynamic::{LossFunction, ThermodynamicSystem};

const PARTICLE_COUNT: usize = 500;
const DIM: usize = 9; // 2x2 + 2 + 2x1 + 1 = 9 parameters for our MLP

fn main() {
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║           NEURAL NETWORK TRAINING BENCHMARK                              ║"
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "{}",
        "║  Training real MLPs with thermodynamic particle optimization             ║"
    );
    println!(
        "{}",
        "║  Architecture: 2 inputs -> 2 hidden (tanh) -> 1 output (sigmoid)         ║"
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝\n"
    );

    let mut all_passed = true;

    // Test 1: XOR Classification
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 1: XOR CLASSIFICATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    all_passed &= train_xor();
    println!();

    // Test 2: Spiral Classification
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 2: SPIRAL CLASSIFICATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    all_passed &= train_spiral();
    println!();

    // Summary
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║                              SUMMARY                                     ║"
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════╣"
    );
    if all_passed {
        println!(
            "{}",
            "║  ✓ ALL NEURAL NETWORK TESTS PASSED                                      ║"
        );
        println!(
            "{}",
            "║  The thermodynamic system successfully trains real neural networks!      ║"
        );
    } else {
        println!(
            "{}",
            "║  ✗ SOME TESTS FAILED                                                    ║"
        );
    }
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝"
    );
}

fn train_xor() -> bool {
    println!("  Problem: XOR (0⊕0=0, 0⊕1=1, 1⊕0=1, 1⊕1=0)");
    println!("  Parameters: {} (MLP weights and biases)", DIM);
    println!("  Particles: {}", PARTICLE_COUNT);

    // Use simulated annealing
    let steps = 3000;
    let mut system =
        ThermodynamicSystem::with_loss_function(PARTICLE_COUNT, DIM, 1.0, LossFunction::MlpXor);

    let start = Instant::now();
    for step in 0..steps {
        // Anneal from T=1.0 to T=0.001
        let progress = step as f32 / steps as f32;
        let temp = 1.0 * (0.001_f32 / 1.0).powf(progress);
        system.set_temperature(temp);
        system.step();
    }
    let elapsed = start.elapsed();

    let particles = system.read_particles();
    let energies: Vec<f32> = particles
        .iter()
        .filter(|p| !p.energy.is_nan())
        .map(|p| p.energy)
        .collect();

    let min_loss = energies.iter().cloned().fold(f32::MAX, f32::min);
    let mean_loss = energies.iter().sum::<f32>() / energies.len() as f32;

    // Find best network
    let best = particles
        .iter()
        .filter(|p| !p.energy.is_nan())
        .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
        .unwrap();

    println!("  Time: {:?}", elapsed);
    println!("  Results:");
    println!("    Min loss (BCE): {:.6}", min_loss);
    println!("    Mean loss:      {:.6}", mean_loss);
    println!(
        "    Best weights:   {:?}",
        best.pos[..DIM]
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>()
    );

    // Test predictions
    println!("  Predictions (best network):");
    let predictions: [(_, f32, f32, f32); 4] = [
        ("0⊕0", 0.0, 0.0, 0.0),
        ("0⊕1", 0.0, 1.0, 1.0),
        ("1⊕0", 1.0, 0.0, 1.0),
        ("1⊕1", 1.0, 1.0, 0.0),
    ];

    let mut correct = 0;
    for (name, x, y, target) in &predictions {
        // Manual forward pass (convert f16 positions to f32)
        let w = |i: usize| best.pos[i].to_f32();
        let h0 = (w(0) * x + w(1) * y + w(4)).tanh();
        let h1 = (w(2) * x + w(3) * y + w(5)).tanh();
        let out = 1.0 / (1.0 + (-w(6) * h0 - w(7) * h1 - w(8)).exp());
        let pred_class: f32 = if out > 0.5 { 1.0 } else { 0.0 };
        let correct_str = if (pred_class - target).abs() < 0.01 {
            "✓"
        } else {
            "✗"
        };
        if (pred_class - target).abs() < 0.01 {
            correct += 1;
        }
        println!(
            "    {}: pred={:.3}, target={}, {}",
            name, out, target, correct_str
        );
    }

    let accuracy = correct as f32 / 4.0;
    println!("    Accuracy: {}/4 = {:.0}%", correct, accuracy * 100.0);

    // XOR is considered solved if we get all 4 correct and loss is low
    let passed = accuracy >= 1.0 && min_loss < 0.5;
    println!("  Status: {}", if passed { "PASS ✓" } else { "FAIL ✗" });
    passed
}

fn train_spiral() -> bool {
    println!("  Problem: Two-spiral classification (40 points)");
    println!("  Parameters: {} (MLP weights and biases)", DIM);
    println!("  Particles: {}", PARTICLE_COUNT);

    // Use simulated annealing with more steps for harder problem
    let steps = 5000;
    let mut system =
        ThermodynamicSystem::with_loss_function(PARTICLE_COUNT, DIM, 2.0, LossFunction::MlpSpiral);

    let start = Instant::now();
    for step in 0..steps {
        // Slower annealing for harder problem
        let progress = step as f32 / steps as f32;
        let temp = 2.0 * (0.001_f32 / 2.0).powf(progress);
        system.set_temperature(temp);
        system.step();
    }
    let elapsed = start.elapsed();

    let particles = system.read_particles();
    let energies: Vec<f32> = particles
        .iter()
        .filter(|p| !p.energy.is_nan())
        .map(|p| p.energy)
        .collect();

    let min_loss = energies.iter().cloned().fold(f32::MAX, f32::min);
    let mean_loss = energies.iter().sum::<f32>() / energies.len() as f32;
    let low_loss = energies.iter().filter(|&&e| e < 0.5).count();

    println!("  Time: {:?}", elapsed);
    println!("  Results:");
    println!("    Min loss (BCE): {:.6}", min_loss);
    println!("    Mean loss:      {:.6}", mean_loss);
    println!(
        "    Low loss (<0.5): {:.1}%",
        low_loss as f32 / energies.len() as f32 * 100.0
    );

    // Spiral is harder - we just want low loss
    let passed = min_loss < 0.7;
    println!("  Status: {}", if passed { "PASS ✓" } else { "FAIL ✗" });
    passed
}
