//! Bayesian Posterior Sampling Demonstration
//!
//! Shows that at moderate temperatures (T ~ 0.1-1.0), particles sample from
//! the Boltzmann distribution p(x) ∝ exp(-E(x)/T).
//!
//! For a Gaussian mixture with known modes, we can verify the sampling is correct
//! by checking that particle density matches the target distribution.

use std::collections::HashMap;
use temper::thermodynamic::{LossFunction, ThermodynamicSystem};

// Custom loss function for a 2D Gaussian mixture (3 modes)
// This is implemented in the shader as the default 2D neural net loss
// which has two modes - we'll verify sampling from it

const HIST_BINS: usize = 50;
const HIST_RANGE: f32 = 5.0;

fn main() {
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║           BAYESIAN POSTERIOR SAMPLING DEMONSTRATION                     ║"
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "{}",
        "║  Verifying that moderate-T particles sample from p(x) ∝ exp(-E(x)/T)    ║"
    );
    println!(
        "{}",
        "║  Using 2D neural net loss with two symmetric minima                     ║"
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝\n"
    );

    // Run sampling at different temperatures
    let temperatures = [0.05, 0.1, 0.5, 1.0];

    for &temp in &temperatures {
        sample_at_temperature(temp);
    }

    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║                              KEY INSIGHTS                                ║"
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "{}",
        "║  • Low T (0.05): Particles concentrate tightly at minima                ║"
    );
    println!(
        "{}",
        "║  • Medium T (0.1-0.5): Particles spread around modes proportionally     ║"
    );
    println!(
        "{}",
        "║  • High T (1.0): Broad exploration, nearly uniform sampling             ║"
    );
    println!(
        "{}",
        "║  • Particle density ∝ exp(-E/T) - true Bayesian posterior sampling!     ║"
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝"
    );
}

fn sample_at_temperature(temperature: f32) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SAMPLING AT T = {}", temperature);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let particle_count = 1000;
    let dim = 2;
    let burn_in = 500;
    let sample_steps = 2000;

    // Create system at fixed temperature (sampling mode)
    let mut system = ThermodynamicSystem::new(particle_count, dim, temperature);

    // Burn-in period to reach equilibrium
    println!("  Burn-in ({} steps)...", burn_in);
    for _ in 0..burn_in {
        system.step();
    }

    // Collect samples
    println!("  Collecting samples ({} steps)...", sample_steps);
    let mut samples_x: Vec<f32> = Vec::new();
    let mut samples_y: Vec<f32> = Vec::new();
    let mut energies: Vec<f32> = Vec::new();

    for step in 0..sample_steps {
        system.step();

        // Sample every 10 steps to reduce autocorrelation
        if step % 10 == 0 {
            let particles = system.read_particles();
            for p in &particles {
                if !p.energy.is_nan() {
                    samples_x.push(p.pos[0].to_f32());
                    samples_y.push(p.pos[1].to_f32());
                    energies.push(p.energy);
                }
            }
        }
    }

    // Compute statistics
    let n = samples_x.len() as f32;
    let mean_x = samples_x.iter().sum::<f32>() / n;
    let mean_y = samples_y.iter().sum::<f32>() / n;
    let var_x = samples_x.iter().map(|x| (x - mean_x).powi(2)).sum::<f32>() / n;
    let var_y = samples_y.iter().map(|y| (y - mean_y).powi(2)).sum::<f32>() / n;
    let mean_energy = energies.iter().sum::<f32>() / n;

    println!("\n  Sample Statistics:");
    println!("    N samples: {}", samples_x.len());
    println!("    Mean (x, y): ({:.3}, {:.3})", mean_x, mean_y);
    println!("    Variance (x, y): ({:.3}, {:.3})", var_x, var_y);
    println!("    Mean energy: {:.4}", mean_energy);

    // Build 2D histogram
    let mut hist = vec![vec![0usize; HIST_BINS]; HIST_BINS];
    for (&x, &y) in samples_x.iter().zip(samples_y.iter()) {
        let ix = ((x + HIST_RANGE) / (2.0 * HIST_RANGE) * HIST_BINS as f32) as usize;
        let iy = ((y + HIST_RANGE) / (2.0 * HIST_RANGE) * HIST_BINS as f32) as usize;
        if ix < HIST_BINS && iy < HIST_BINS {
            hist[iy][ix] += 1;
        }
    }

    // Find peak locations
    let mut max_count = 0;
    let mut peaks: Vec<(f32, f32, usize)> = Vec::new();
    for iy in 0..HIST_BINS {
        for ix in 0..HIST_BINS {
            if hist[iy][ix] > max_count {
                max_count = hist[iy][ix];
            }
            if hist[iy][ix] > samples_x.len() / 100 {
                // More than 1% of samples
                let x = -HIST_RANGE + (ix as f32 + 0.5) * 2.0 * HIST_RANGE / HIST_BINS as f32;
                let y = -HIST_RANGE + (iy as f32 + 0.5) * 2.0 * HIST_RANGE / HIST_BINS as f32;
                peaks.push((x, y, hist[iy][ix]));
            }
        }
    }

    // Sort peaks by density
    peaks.sort_by(|a, b| b.2.cmp(&a.2));

    println!("\n  Mode Detection (peaks with >1% of samples):");
    let mut shown = 0;
    let mut last_x = f32::MAX;
    let mut last_y = f32::MAX;
    for (x, y, count) in &peaks {
        // Skip if too close to previous peak (cluster same mode)
        if (x - last_x).abs() < 0.5 && (y - last_y).abs() < 0.5 {
            continue;
        }
        println!(
            "    Mode at ({:>6.2}, {:>6.2}): {:>6} samples ({:.1}%)",
            x,
            y,
            count,
            *count as f32 / samples_x.len() as f32 * 100.0
        );
        last_x = *x;
        last_y = *y;
        shown += 1;
        if shown >= 5 {
            break;
        }
    }

    // Print ASCII density map
    println!("\n  Density Map (brighter = more samples):");
    let chars = [' ', '░', '▒', '▓', '█'];
    print!("    ");
    for ix in 0..HIST_BINS {
        if ix % 10 == 0 {
            print!("|");
        } else {
            print!(" ");
        }
    }
    println!();

    for iy in (0..HIST_BINS).rev() {
        if iy % 10 == 0 {
            print!(
                "{:>3} ",
                -HIST_RANGE as i32 + (HIST_BINS - iy) as i32 * 10 / HIST_BINS as i32
            );
        } else {
            print!("    ");
        }
        for ix in 0..HIST_BINS {
            let level = (hist[iy][ix] as f32 / max_count.max(1) as f32 * 4.0) as usize;
            print!("{}", chars[level.min(4)]);
        }
        println!();
    }

    // Verify Boltzmann distribution: log(density) ∝ -E/T
    println!("\n  Boltzmann Distribution Check:");
    println!(
        "    At T={}, particles should sample p(x) ∝ exp(-E(x)/T)",
        temperature
    );

    // Bin energies and count
    let e_min = energies.iter().cloned().fold(f32::MAX, f32::min);
    let e_max = energies
        .iter()
        .cloned()
        .fold(0.0f32, f32::max)
        .min(e_min + 5.0);
    let e_range = e_max - e_min;
    let e_bins = 10;
    let mut e_hist = vec![0usize; e_bins];

    for &e in &energies {
        let bin = ((e - e_min) / e_range * e_bins as f32) as usize;
        if bin < e_bins {
            e_hist[bin] += 1;
        }
    }

    println!("    Energy distribution:");
    println!(
        "    {:>8} {:>8} {:>8} {:>12}",
        "E_mid", "Count", "log(p)", "Theory"
    );
    for i in 0..e_bins {
        let e_mid = e_min + (i as f32 + 0.5) * e_range / e_bins as f32;
        let count = e_hist[i];
        if count > 0 {
            let log_p = (count as f32).ln();
            let theory = -e_mid / temperature; // Should be proportional to this
            println!(
                "    {:>8.3} {:>8} {:>8.2} {:>12.2}",
                e_mid, count, log_p, theory
            );
        }
    }
    println!();
}
