//! Parallel Tempering (Replica Exchange) Implementation
//!
//! Runs multiple particle systems at different temperatures simultaneously.
//! Periodically swaps configurations between adjacent temperature levels
//! using Metropolis-Hastings acceptance criteria.
//!
//! Benefits:
//! - Hot replicas explore broadly, finding new basins
//! - Cold replicas exploit locally, refining solutions
//! - Swaps allow good solutions to propagate to cold replicas
//! - Much better at escaping local minima than single-temperature runs

use temper::thermodynamic::{LossFunction, ThermodynamicSystem, ThermodynamicParticle};
use std::time::Instant;

/// Parallel tempering system with multiple temperature replicas
pub struct ParallelTempering {
    replicas: Vec<ThermodynamicSystem>,
    temperatures: Vec<f32>,
    swap_interval: usize,
    step_count: usize,
    total_swaps: usize,
    accepted_swaps: usize,
}

impl ParallelTempering {
    /// Create a new parallel tempering system
    pub fn new(
        num_replicas: usize,
        particle_count: usize,
        dim: usize,
        temp_min: f32,
        temp_max: f32,
        loss_fn: LossFunction,
        swap_interval: usize,
    ) -> Self {
        // Create geometric temperature ladder
        let mut temperatures = Vec::with_capacity(num_replicas);
        for i in 0..num_replicas {
            let t = if num_replicas == 1 {
                temp_min
            } else {
                let ratio = i as f32 / (num_replicas - 1) as f32;
                temp_min * (temp_max / temp_min).powf(ratio)
            };
            temperatures.push(t);
        }

        // Create replicas at each temperature
        let replicas = temperatures.iter()
            .map(|&t| ThermodynamicSystem::with_loss_function(particle_count, dim, t, loss_fn))
            .collect();

        Self {
            replicas,
            temperatures,
            swap_interval,
            step_count: 0,
            total_swaps: 0,
            accepted_swaps: 0,
        }
    }

    /// Run one step of all replicas, with potential replica exchange
    pub fn step(&mut self) {
        self.step_count += 1;

        // Step all replicas
        for replica in &mut self.replicas {
            replica.step();
        }

        // Attempt replica exchange every swap_interval steps
        if self.step_count % self.swap_interval == 0 {
            self.attempt_swaps();
        }
    }

    /// Attempt swaps between adjacent temperature levels
    fn attempt_swaps(&mut self) {
        let n = self.replicas.len();
        if n < 2 {
            return;
        }

        // Alternate between even-odd and odd-even pairings
        let offset = if (self.step_count / self.swap_interval) % 2 == 0 { 0 } else { 1 };

        let mut i = offset;
        while i + 1 < n {
            self.total_swaps += 1;

            // Get best energies from each replica
            let particles_i = self.replicas[i].read_particles();
            let particles_j = self.replicas[i + 1].read_particles();

            let e_i = particles_i.iter()
                .filter(|p| !p.energy.is_nan())
                .map(|p| p.energy)
                .fold(f32::MAX, f32::min);
            let e_j = particles_j.iter()
                .filter(|p| !p.energy.is_nan())
                .map(|p| p.energy)
                .fold(f32::MAX, f32::min);

            let t_i = self.temperatures[i];
            let t_j = self.temperatures[i + 1];

            // Metropolis-Hastings acceptance criterion for replica exchange
            // Δ = (1/T_i - 1/T_j) * (E_j - E_i)
            let delta = (1.0 / t_i - 1.0 / t_j) * (e_j - e_i);
            let accept = delta < 0.0 || rand_float() < (-delta).exp();

            if accept {
                self.accepted_swaps += 1;
                // Swap temperatures (effectively swaps which system runs at which T)
                self.replicas[i].set_temperature(t_j);
                self.replicas[i + 1].set_temperature(t_i);
                self.temperatures.swap(i, i + 1);
            }

            i += 2;
        }
    }

    /// Get the coldest replica (best for optimization)
    pub fn coldest_replica(&self) -> &ThermodynamicSystem {
        let min_idx = self.temperatures.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        &self.replicas[min_idx]
    }

    /// Get best particle across all replicas
    pub fn best_particle(&self) -> ThermodynamicParticle {
        let mut best: Option<ThermodynamicParticle> = None;
        let mut best_energy = f32::MAX;

        for replica in &self.replicas {
            let particles = replica.read_particles();
            for p in particles {
                if !p.energy.is_nan() && p.energy < best_energy {
                    best_energy = p.energy;
                    best = Some(p);
                }
            }
        }

        best.unwrap_or_else(|| ThermodynamicParticle {
            pos: [0.0; 64],
            energy: f32::MAX,
            entropy_bits: 0,
            _pad: [0.0; 2],
        })
    }

    /// Get statistics about the system
    pub fn stats(&self) -> ParallelTemperingStats {
        let replica_stats: Vec<(f32, f32, f32)> = self.replicas.iter()
            .zip(&self.temperatures)
            .map(|(replica, &temp)| {
                let particles = replica.read_particles();
                let energies: Vec<f32> = particles.iter()
                    .filter(|p| !p.energy.is_nan())
                    .map(|p| p.energy)
                    .collect();
                let min = energies.iter().cloned().fold(f32::MAX, f32::min);
                let mean = energies.iter().sum::<f32>() / energies.len().max(1) as f32;
                (temp, min, mean)
            })
            .collect();

        let swap_rate = if self.total_swaps > 0 {
            self.accepted_swaps as f32 / self.total_swaps as f32
        } else {
            0.0
        };

        ParallelTemperingStats {
            replica_stats,
            swap_rate,
            total_swaps: self.total_swaps,
            accepted_swaps: self.accepted_swaps,
        }
    }
}

#[derive(Debug)]
pub struct ParallelTemperingStats {
    pub replica_stats: Vec<(f32, f32, f32)>,  // (temp, min_energy, mean_energy)
    pub swap_rate: f32,
    pub total_swaps: usize,
    pub accepted_swaps: usize,
}

// Simple random number for swap decisions
fn rand_float() -> f32 {
    use std::time::SystemTime;
    let t = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
    let x = t.subsec_nanos() as u32;
    let x = x ^ (x >> 16);
    let x = x.wrapping_mul(0x7feb352d);
    let x = x ^ (x >> 15);
    (x & 0xFFFFFF) as f32 / 16777216.0
}

fn main() {
    println!("{}",
        "╔══════════════════════════════════════════════════════════════════════════╗");
    println!("{}",
        "║           PARALLEL TEMPERING DEMONSTRATION                              ║");
    println!("{}",
        "╠══════════════════════════════════════════════════════════════════════════╣");
    println!("{}",
        "║  Running multiple temperature replicas with replica exchange            ║");
    println!("{}",
        "║  Hot replicas explore, cold replicas exploit, swaps propagate solutions ║");
    println!("{}",
        "╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Test on Rastrigin - highly multimodal, hard for single-temperature methods
    test_rastrigin();

    // Test on deep MLP
    test_deep_mlp();
}

fn test_rastrigin() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST: RASTRIGIN 4D with Parallel Tempering");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let num_replicas = 6;
    let particle_count = 200;
    let dim = 4;
    let steps = 3000;

    println!("  Replicas: {}", num_replicas);
    println!("  Particles per replica: {}", particle_count);
    println!("  Temperature range: 0.001 to 2.0");
    println!("  Swap interval: every 50 steps");

    let mut pt = ParallelTempering::new(
        num_replicas,
        particle_count,
        dim,
        0.001,  // Cold
        2.0,    // Hot
        LossFunction::Rastrigin,
        50,
    );

    let start = Instant::now();
    for step in 0..steps {
        pt.step();

        if step % 1000 == 0 && step > 0 {
            let best = pt.best_particle();
            println!("    Step {}: best_loss = {:.4}", step, best.energy);
        }
    }
    let elapsed = start.elapsed();

    let stats = pt.stats();
    let best = pt.best_particle();

    println!("\n  Results:");
    println!("    Time: {:?}", elapsed);
    println!("    Best loss: {:.6}", best.energy);
    println!("    Best position: {:?}", &best.pos[..dim]);
    println!("    Swap rate: {:.1}% ({}/{})",
        stats.swap_rate * 100.0, stats.accepted_swaps, stats.total_swaps);

    println!("\n  Per-replica statistics:");
    println!("    {:>10} {:>12} {:>12}", "Temp", "Min Loss", "Mean Loss");
    for (temp, min, mean) in &stats.replica_stats {
        println!("    {:>10.4} {:>12.4} {:>12.4}", temp, min, mean);
    }

    // Compare to single temperature
    println!("\n  Comparison to single-temperature run:");
    let mut single = ThermodynamicSystem::with_loss_function(
        particle_count * num_replicas, dim, 0.001, LossFunction::Rastrigin
    );
    for step in 0..steps {
        let progress = step as f32 / steps as f32;
        let temp = 1.0 * (0.001_f32 / 1.0).powf(progress);
        single.set_temperature(temp);
        single.step();
    }
    let single_particles = single.read_particles();
    let single_best = single_particles.iter()
        .filter(|p| !p.energy.is_nan())
        .map(|p| p.energy)
        .fold(f32::MAX, f32::min);

    println!("    Parallel tempering best: {:.6}", best.energy);
    println!("    Single annealing best:   {:.6}", single_best);
    println!("    Winner: {}", if best.energy < single_best { "PARALLEL TEMPERING" } else { "SINGLE" });
}

fn test_deep_mlp() {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST: DEEP MLP (37 params) with Parallel Tempering");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let num_replicas = 4;
    let particle_count = 250;
    let dim = 37;
    let steps = 5000;

    println!("  Replicas: {}", num_replicas);
    println!("  Particles per replica: {}", particle_count);
    println!("  Temperature range: 0.0001 to 2.0");

    let mut pt = ParallelTempering::new(
        num_replicas,
        particle_count,
        dim,
        0.0001,
        2.0,
        LossFunction::MlpDeep,
        100,
    );

    let start = Instant::now();
    for step in 0..steps {
        pt.step();

        if step % 1000 == 0 && step > 0 {
            let best = pt.best_particle();
            println!("    Step {}: best_loss = {:.4}", step, best.energy);
        }
    }
    let elapsed = start.elapsed();

    let stats = pt.stats();
    let best = pt.best_particle();

    println!("\n  Results:");
    println!("    Time: {:?}", elapsed);
    println!("    Best loss (BCE): {:.6}", best.energy);
    println!("    Swap rate: {:.1}%", stats.swap_rate * 100.0);

    let passed = best.energy < 0.5;
    println!("\n  Status: {}", if passed { "PASS ✓" } else { "FAIL ✗" });
}
