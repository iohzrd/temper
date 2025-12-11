//! Thermodynamic Entropy Stream
//!
//! Outputs a continuous stream of entropy bytes from the thermodynamic particle system
//! running in high-temperature (entropy) mode.
//!
//! Usage:
//!   cargo run --release --features gpu --bin thermodynamic-stream | dieharder -a -g 200
//!   cargo run --release --features gpu --bin thermodynamic-stream | ent
//!
//! This validates that the unified thermodynamic system produces cryptographic-quality
//! randomness when operating in entropy mode (T >> 1).

use nbody_entropy::thermodynamic::ThermodynamicSystem;
use std::io::{self, Write};

const PARTICLE_COUNT: usize = 1000;
const DIM: usize = 2;
const TEMPERATURE: f32 = 10.0; // High temperature for entropy mode

fn main() {
    eprintln!("Thermodynamic Entropy Stream");
    eprintln!("  Particles: {}", PARTICLE_COUNT);
    eprintln!("  Dimensions: {}", DIM);
    eprintln!("  Temperature: {} (ENTROPY mode)", TEMPERATURE);
    eprintln!("  Streaming to stdout...");
    eprintln!();

    let mut system = ThermodynamicSystem::new(PARTICLE_COUNT, DIM, TEMPERATURE);

    // Warmup
    for _ in 0..100 {
        system.step();
    }

    let mut stdout = io::stdout().lock();
    let mut total_bytes = 0u64;

    loop {
        // Run a step
        system.step();

        // Extract entropy
        let entropy = system.extract_entropy();

        // Convert to bytes and write to stdout
        for val in entropy {
            let bytes = val.to_le_bytes();
            if stdout.write_all(&bytes).is_err() {
                // Pipe closed (e.g., dieharder finished)
                eprintln!("\nStream ended. Total bytes: {}", total_bytes);
                return;
            }
            total_bytes += 4;
        }

        // Flush periodically
        if total_bytes % (1024 * 1024) < 4000 {
            let _ = stdout.flush();
        }
    }
}
