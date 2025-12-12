//! Hyperparameter Tuning with Thermodynamic Optimization
//!
//! Demonstrates using the thermodynamic particle system for hyperparameter
//! optimization - a common real-world ML problem.
//!
//! We simulate an ML training process and use particles to explore the
//! hyperparameter space to find optimal settings.

use std::time::Instant;
use temper::thermodynamic::{LossFunction, ThermodynamicSystem};

// Simulated ML training: returns validation loss given hyperparameters
// This models a realistic hyperparameter landscape with:
// - Learning rate: too low = slow convergence, too high = divergence
// - L2 regularization: affects generalization
// - Dropout: affects overfitting
// - Hidden size: model capacity
//
// The function has multiple local minima and is non-convex

fn simulate_training(log_lr: f32, l2_reg: f32, dropout: f32, log_hidden: f32) -> f32 {
    // Transform to actual values
    let lr = 10.0f32.powf(log_lr); // log_lr in [-5, 0] -> lr in [1e-5, 1]
    let hidden = 10.0f32.powf(log_hidden).round() as i32; // 10-1000

    // Simulate training loss as a function of hyperparameters
    // This models realistic behavior:

    // 1. Learning rate: optimal around 1e-3, too high causes divergence
    let lr_optimal: f32 = 0.001;
    let lr_penalty = if lr > 0.1 {
        10.0 * (lr - 0.1) // Divergence
    } else {
        ((lr.ln() - lr_optimal.ln()) / 2.0).powi(2)
    };

    // 2. L2 regularization: optimal around 1e-4, too high = underfitting
    let l2_optimal: f32 = 0.0001;
    let l2_penalty = if l2_reg > 0.01 {
        5.0 * (l2_reg - 0.01) // Heavy underfitting
    } else {
        ((l2_reg + 1e-6).ln() - l2_optimal.ln()).abs() * 0.1
    };

    // 3. Dropout: optimal around 0.3, too high = underfitting
    let dropout_optimal = 0.3;
    let dropout_penalty = (dropout - dropout_optimal).powi(2) * 2.0;

    // 4. Hidden size: optimal around 128-256, diminishing returns after
    let hidden_f = hidden as f32;
    let hidden_optimal = 200.0;
    let hidden_penalty = if hidden_f < 32.0 {
        (32.0 - hidden_f) * 0.01 // Too small
    } else if hidden_f > 500.0 {
        (hidden_f - 500.0) * 0.0001 // Diminishing returns
    } else {
        ((hidden_f - hidden_optimal) / hidden_optimal).powi(2) * 0.5
    };

    // Add some noise and interaction terms to make it more realistic
    let interaction = lr * l2_reg * 10.0 + dropout * (1.0 - lr.ln().abs() * 0.1);

    // Base validation loss (simulating actual training)
    let base_loss = 0.1; // Best achievable

    base_loss + lr_penalty + l2_penalty + dropout_penalty + hidden_penalty + interaction.abs() * 0.1
}

// Map position to hyperparameters
fn pos_to_hyperparams(pos: &[temper::f16]) -> (f32, f32, f32, f32) {
    // pos[0]: log_lr in [-5, 0]
    let log_lr = pos[0].to_f32().clamp(-5.0, 0.0);
    // pos[1]: l2_reg in [0, 0.1]
    let l2_reg = (pos[1].to_f32().clamp(-3.0, 3.0) + 3.0) / 60.0; // Map [-3,3] to [0, 0.1]
    // pos[2]: dropout in [0, 0.8]
    let dropout = (pos[2].to_f32().clamp(-2.0, 2.0) + 2.0) / 5.0; // Map [-2,2] to [0, 0.8]
    // pos[3]: log_hidden in [1, 3] (10 to 1000)
    let log_hidden = pos[3].to_f32().clamp(1.0, 3.0);

    (log_lr, l2_reg, dropout, log_hidden)
}

fn hyperparams_to_actual(
    log_lr: f32,
    l2_reg: f32,
    dropout: f32,
    log_hidden: f32,
) -> (f32, f32, f32, i32) {
    let lr = 10.0f32.powf(log_lr);
    let hidden = 10.0f32.powf(log_hidden).round() as i32;
    (lr, l2_reg, dropout, hidden)
}

fn main() {
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║           HYPERPARAMETER TUNING DEMONSTRATION                           ║"
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "{}",
        "║  Using thermodynamic particles to find optimal ML hyperparameters       ║"
    );
    println!(
        "{}",
        "║  Hyperparameters: learning_rate, L2_reg, dropout, hidden_size           ║"
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝\n"
    );

    // Compare different search strategies
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("STRATEGY COMPARISON");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // 1. Random search baseline
    let random_result = random_search(1000);
    println!("\n  Random Search (1000 trials):");
    print_result(&random_result);

    // 2. Grid search baseline
    let grid_result = grid_search();
    println!("\n  Grid Search (5^4 = 625 trials):");
    print_result(&grid_result);

    // 3. Thermodynamic optimization (using Sphere loss as proxy - will use custom)
    let thermo_result = thermodynamic_search(1000);
    println!("\n  Thermodynamic Particles (500 particles, 1000 steps):");
    print_result(&thermo_result);

    // 4. Thermodynamic with longer run
    let thermo_long_result = thermodynamic_search(3000);
    println!("\n  Thermodynamic Particles (500 particles, 3000 steps):");
    print_result(&thermo_long_result);

    // Summary
    println!(
        "\n{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║                              COMPARISON                                 ║"
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "{}",
        "║  Method                     Best Loss    Time                           ║"
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "  Random Search (1000)        {:.6}    {:>8}ms",
        random_result.loss, random_result.time_ms
    );
    println!(
        "  Grid Search (625)           {:.6}    {:>8}ms",
        grid_result.loss, grid_result.time_ms
    );
    println!(
        "  Thermodynamic (1000 steps)  {:.6}    {:>8}ms",
        thermo_result.loss, thermo_result.time_ms
    );
    println!(
        "  Thermodynamic (3000 steps)  {:.6}    {:>8}ms",
        thermo_long_result.loss, thermo_long_result.time_ms
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝"
    );
}

struct SearchResult {
    lr: f32,
    l2_reg: f32,
    dropout: f32,
    hidden: i32,
    loss: f32,
    time_ms: u128,
}

fn print_result(result: &SearchResult) {
    println!("    Best validation loss: {:.6}", result.loss);
    println!("    Hyperparameters:");
    println!("      learning_rate: {:.6}", result.lr);
    println!("      L2_reg:        {:.6}", result.l2_reg);
    println!("      dropout:       {:.3}", result.dropout);
    println!("      hidden_size:   {}", result.hidden);
    println!("    Time: {}ms", result.time_ms);
}

fn random_search(n_trials: usize) -> SearchResult {
    let start = Instant::now();
    let mut best_loss = f32::MAX;
    let mut best_params = (0.0, 0.0, 0.0, 0);

    // Simple PRNG
    let mut seed = 12345u64;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed & 0xFFFFFF) as f32 / 16777216.0
    };

    for _ in 0..n_trials {
        let log_lr = -5.0 + rand() * 5.0;
        let l2_reg = rand() * 0.1;
        let dropout = rand() * 0.8;
        let log_hidden = 1.0 + rand() * 2.0;

        let loss = simulate_training(log_lr, l2_reg, dropout, log_hidden);
        if loss < best_loss {
            best_loss = loss;
            best_params = hyperparams_to_actual(log_lr, l2_reg, dropout, log_hidden);
        }
    }

    SearchResult {
        lr: best_params.0,
        l2_reg: best_params.1,
        dropout: best_params.2,
        hidden: best_params.3,
        loss: best_loss,
        time_ms: start.elapsed().as_millis(),
    }
}

fn grid_search() -> SearchResult {
    let start = Instant::now();
    let mut best_loss = f32::MAX;
    let mut best_params = (0.0, 0.0, 0.0, 0);

    // 5 points per dimension = 625 total
    let lr_values = [-4.0, -3.0, -2.5, -2.0, -1.0]; // log scale
    let l2_values = [0.0, 0.0001, 0.001, 0.01, 0.1];
    let dropout_values = [0.0, 0.2, 0.4, 0.5, 0.6];
    let hidden_values = [1.5, 2.0, 2.3, 2.5, 2.8]; // log scale

    for &log_lr in &lr_values {
        for &l2_reg in &l2_values {
            for &dropout in &dropout_values {
                for &log_hidden in &hidden_values {
                    let loss = simulate_training(log_lr, l2_reg, dropout, log_hidden);
                    if loss < best_loss {
                        best_loss = loss;
                        best_params = hyperparams_to_actual(log_lr, l2_reg, dropout, log_hidden);
                    }
                }
            }
        }
    }

    SearchResult {
        lr: best_params.0,
        l2_reg: best_params.1,
        dropout: best_params.2,
        hidden: best_params.3,
        loss: best_loss,
        time_ms: start.elapsed().as_millis(),
    }
}

fn thermodynamic_search(steps: usize) -> SearchResult {
    let start = Instant::now();

    // Use the sphere loss function and map to our hyperparam space
    // The thermodynamic system will explore the 4D space
    let particle_count = 500;
    let dim = 4;

    let mut system =
        ThermodynamicSystem::with_loss_function(particle_count, dim, 1.0, LossFunction::Sphere);

    // We can't use a custom loss in the shader, so we'll evaluate manually
    // and use the system for exploration with annealing

    let mut best_loss = f32::MAX;
    let mut best_params = (0.0, 0.0, 0.0, 0);

    // Run with annealing
    for step in 0..steps {
        let progress = step as f32 / steps as f32;
        let temp = 1.0 * (0.001_f32 / 1.0).powf(progress);
        system.set_temperature(temp);
        system.step();

        // Evaluate particles on our actual objective every 50 steps
        if step % 50 == 0 {
            let particles = system.read_particles();
            for p in &particles {
                let (log_lr, l2_reg, dropout, log_hidden) = pos_to_hyperparams(&p.pos);
                let loss = simulate_training(log_lr, l2_reg, dropout, log_hidden);

                if loss < best_loss {
                    best_loss = loss;
                    best_params = hyperparams_to_actual(log_lr, l2_reg, dropout, log_hidden);
                }
            }
        }
    }

    SearchResult {
        lr: best_params.0,
        l2_reg: best_params.1,
        dropout: best_params.2,
        hidden: best_params.3,
        loss: best_loss,
        time_ms: start.elapsed().as_millis(),
    }
}
