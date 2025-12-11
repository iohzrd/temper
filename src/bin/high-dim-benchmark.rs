//! High-Dimensional Adaptive vs Fixed Annealing Benchmark
//!
//! Tests whether adaptive scheduling advantage grows with dimensionality.
//! Higher dimensions = more local minima = harder to escape = adaptive should help more.
//!
//! Run with: cargo run --release --features gpu --bin high-dim-benchmark

use nbody_entropy::{AdaptiveScheduler, LossFunction, ThermodynamicSystem};
use std::time::Instant;

const PARTICLE_COUNT: usize = 500;
const STEPS: u32 = 5000;
const TRIALS: usize = 5;

struct TrialResult {
    final_energy: f32,
    steps_to_threshold: Option<u32>,
    reheats: u32,
    elapsed_ms: u128,
}

fn run_fixed(dim: usize, t_start: f32, t_end: f32) -> TrialResult {
    let start = Instant::now();
    let mut system = ThermodynamicSystem::with_loss_function(
        PARTICLE_COUNT, dim, t_start, LossFunction::Rastrigin
    );
    system.set_repulsion_samples(64);

    let threshold = 0.1 * dim as f32; // Scale threshold with dimension
    let mut steps_to_threshold = None;
    let mut min_energy = f32::MAX;

    for step in 0..STEPS {
        let progress = step as f32 / STEPS as f32;
        let temp = t_start * (t_end / t_start).powf(progress);
        system.set_temperature(temp);

        let dt = if temp > 0.1 { 0.01 } else if temp > 0.01 { 0.005 } else { 0.002 };
        system.set_dt(dt);

        system.step();

        // Check energy
        let particles = system.read_particles();
        for p in &particles {
            if !p.energy.is_nan() && p.energy < min_energy {
                min_energy = p.energy;
            }
        }

        if steps_to_threshold.is_none() && min_energy < threshold {
            steps_to_threshold = Some(step);
        }
    }

    TrialResult {
        final_energy: min_energy,
        steps_to_threshold,
        reheats: 0,
        elapsed_ms: start.elapsed().as_millis(),
    }
}

fn run_adaptive(dim: usize, t_start: f32, t_end: f32) -> TrialResult {
    let start = Instant::now();
    let mut system = ThermodynamicSystem::with_loss_function(
        PARTICLE_COUNT, dim, t_start, LossFunction::Rastrigin
    );
    system.set_repulsion_samples(64);

    let threshold = 0.1 * dim as f32;
    let mut scheduler = AdaptiveScheduler::with_steps(t_start, t_end, threshold, dim, STEPS);
    let mut steps_to_threshold = None;
    let mut min_energy = f32::MAX;

    for step in 0..STEPS {
        let temp = scheduler.update(min_energy);
        system.set_temperature(temp);

        let dt = if temp > 0.1 { 0.01 } else if temp > 0.01 { 0.005 } else { 0.002 };
        system.set_dt(dt);

        system.step();

        let particles = system.read_particles();
        for p in &particles {
            if !p.energy.is_nan() && p.energy < min_energy {
                min_energy = p.energy;
            }
        }

        if steps_to_threshold.is_none() && min_energy < threshold {
            steps_to_threshold = Some(step);
        }
    }

    TrialResult {
        final_energy: min_energy,
        steps_to_threshold,
        reheats: scheduler.reheat_count(),
        elapsed_ms: start.elapsed().as_millis(),
    }
}

fn main() {
    println!("High-Dimensional Adaptive vs Fixed Annealing Benchmark");
    println!("=======================================================");
    println!("Function: Rastrigin (global min = 0 at origin)");
    println!("Particles: {}, Steps: {}, Trials: {}", PARTICLE_COUNT, STEPS, TRIALS);
    println!();

    let dimensions = [2, 4, 8, 16, 32];

    for &dim in &dimensions {
        println!("--- Dimension: {} ---", dim);

        // Scale temperature with dimension
        let t_start = 5.0 * (dim as f32 / 2.0).sqrt();
        let t_end = 0.001;
        let threshold = 0.1 * dim as f32;

        let mut fixed_results: Vec<TrialResult> = Vec::new();
        let mut adaptive_results: Vec<TrialResult> = Vec::new();

        print!("Running fixed...    ");
        for _ in 0..TRIALS {
            fixed_results.push(run_fixed(dim, t_start, t_end));
            print!(".");
        }
        println!(" done");

        print!("Running adaptive... ");
        for _ in 0..TRIALS {
            adaptive_results.push(run_adaptive(dim, t_start, t_end));
            print!(".");
        }
        println!(" done");

        // Compute statistics
        let fixed_energies: Vec<f32> = fixed_results.iter().map(|r| r.final_energy).collect();
        let adaptive_energies: Vec<f32> = adaptive_results.iter().map(|r| r.final_energy).collect();

        let fixed_mean = fixed_energies.iter().sum::<f32>() / TRIALS as f32;
        let fixed_min = fixed_energies.iter().cloned().fold(f32::MAX, f32::min);
        let fixed_max = fixed_energies.iter().cloned().fold(f32::MIN, f32::max);

        let adaptive_mean = adaptive_energies.iter().sum::<f32>() / TRIALS as f32;
        let adaptive_min = adaptive_energies.iter().cloned().fold(f32::MAX, f32::min);
        let adaptive_max = adaptive_energies.iter().cloned().fold(f32::MIN, f32::max);

        let fixed_success = fixed_results.iter().filter(|r| r.steps_to_threshold.is_some()).count();
        let adaptive_success = adaptive_results.iter().filter(|r| r.steps_to_threshold.is_some()).count();

        let adaptive_reheats: u32 = adaptive_results.iter().map(|r| r.reheats).sum();

        let fixed_conv_steps: Vec<u32> = fixed_results.iter()
            .filter_map(|r| r.steps_to_threshold)
            .collect();
        let adaptive_conv_steps: Vec<u32> = adaptive_results.iter()
            .filter_map(|r| r.steps_to_threshold)
            .collect();

        let fixed_avg_steps = if !fixed_conv_steps.is_empty() {
            fixed_conv_steps.iter().sum::<u32>() as f32 / fixed_conv_steps.len() as f32
        } else { f32::NAN };

        let adaptive_avg_steps = if !adaptive_conv_steps.is_empty() {
            adaptive_conv_steps.iter().sum::<u32>() as f32 / adaptive_conv_steps.len() as f32
        } else { f32::NAN };

        println!();
        println!("  FIXED:    mean={:8.2}  min={:8.2}  max={:8.2}  success={}/{}  avg_steps={:.0}",
                 fixed_mean, fixed_min, fixed_max, fixed_success, TRIALS, fixed_avg_steps);
        println!("  ADAPTIVE: mean={:8.2}  min={:8.2}  max={:8.2}  success={}/{}  avg_steps={:.0}  reheats={}",
                 adaptive_mean, adaptive_min, adaptive_max, adaptive_success, TRIALS, adaptive_avg_steps, adaptive_reheats);

        let winner = if adaptive_mean < fixed_mean { "ADAPTIVE" } else { "FIXED" };
        let margin = (fixed_mean - adaptive_mean).abs();
        println!("  Winner: {} by {:.2}", winner, margin);
        println!();
    }

    println!("Benchmark complete.");
}
