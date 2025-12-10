//! # N-body Entropy
//!
//! Experimental entropy extraction from N-body particle simulation through
//! self-referential feedback loops and chaotic gravitational dynamics.
//!
//! ## Concept
//!
//! Traditional PRNGs use mathematical operations (xorshift, LCG, etc.) to produce
//! pseudo-random sequences. This library explores a different approach: using a
//! **high-dimensional physical simulation** as the entropy source.
//!
//! The key insight: a feedback loop where **outputs determine which parts of the
//! system to query next**, combined with **chaotic n-body dynamics**, creates
//! compounding complexity that becomes practically unpredictable.
//!
//! ## Example
//!
//! ```rust
//! use nbody_entropy::NbodyEntropy;
//! use rand_core::RngCore;
//!
//! let mut rng = NbodyEntropy::new();
//!
//! // Generate random values
//! let value = rng.next_u64();
//!
//! // Fill a buffer
//! let mut buf = [0u8; 32];
//! rng.fill_bytes(&mut buf);
//! ```

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub use gpu::GpuNbodyEntropy;

use rand_core::{RngCore, SeedableRng};
use std::f64::consts::PI;
use std::time::{SystemTime, UNIX_EPOCH};

/// Default number of particles in the simulation
pub const DEFAULT_PARTICLE_COUNT: usize = 256;

/// Maximum supported particles
pub const MAX_PARTICLE_COUNT: usize = 1024;

/// Number of attractor particles (these interact gravitationally)
pub const ATTRACTOR_COUNT: usize = 12;

/// Gravitational constant for n-body simulation
const G: f64 = 0.0001;

/// Softening parameter to prevent singularities
const SOFTENING: f64 = 0.001;

/// Damping factor to prevent runaway velocities
const DAMPING: f64 = 0.999;

/// Boundary size (particles wrap around)
const BOUNDARY: f64 = 1.0;

/// Fast acos approximation (Corrsin's approximation, ~0.5% max error)
#[inline(always)]
fn fast_acos(x: f64) -> f64 {
    let x = x.clamp(-1.0, 1.0);
    let abs_x = x.abs();
    let result = (-0.0187293 * abs_x + 0.0742610) * abs_x - 0.2121144;
    let result = (result * abs_x + 1.5707288) * (1.0 - abs_x).sqrt();
    if x < 0.0 { PI - result } else { result }
}

/// Wrap position to stay within bounds
#[inline(always)]
fn wrap(x: f64) -> f64 {
    let x = x % BOUNDARY;
    if x < 0.0 { x + BOUNDARY } else { x }
}

/// A particle with position and velocity (for n-body dynamics)
#[derive(Clone, Debug)]
struct Particle {
    /// Current position
    pos: [f64; 2],
    /// Current velocity
    vel: [f64; 2],
    /// Mass (attractors have higher mass)
    mass: f64,
    /// Is this an attractor particle?
    is_attractor: bool,
}

impl Particle {
    /// Get current position
    #[inline(always)]
    fn position(&self) -> [f64; 2] {
        self.pos
    }
}

/// Query types for extracting relational data from the particle system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Euclidean distance between two particles
    Distance,
    /// Angle formed by three particles at vertex j
    Angle,
    /// Count of particles within threshold distance
    Neighbors,
    /// Product of two distances
    CombinedDistance,
    /// Graph connectivity: are i and j in the same connected component?
    Connectivity,
    /// Velocity magnitude of particle i
    Velocity,
}

impl QueryType {
    /// Get query type from index - weight towards faster queries
    #[inline(always)]
    fn from_index(idx: u64) -> Self {
        match idx % 16 {
            0..=5 => QueryType::Distance,           // 6/16 = 37.5%
            6..=10 => QueryType::Angle,             // 5/16 = 31.25%
            11..=12 => QueryType::CombinedDistance, // 2/16 = 12.5%
            13 => QueryType::Neighbors,             // 1/16 = 6.25%
            14 => QueryType::Connectivity,          // 1/16 = 6.25%
            _ => QueryType::Velocity,               // 1/16 = 6.25%
        }
    }
}

/// The entropy-generating system
///
/// Uses a particle simulation with n-body dynamics and feedback loops to generate
/// pseudo-random numbers. Attractor particles interact gravitationally (chaotic),
/// while follower particles are influenced by attractors.
#[derive(Clone)]
pub struct NbodyEntropy {
    particles: Vec<Particle>,
    particle_count: usize,
    /// Internal state mixed with query outputs via BLAKE3
    state: [u8; 32],
    /// Counter for additional entropy
    counter: u64,
    /// Connection threshold for graph queries
    connection_threshold: f64,
    /// BLAKE3 hasher key
    hasher_key: [u8; 32],
    /// Time step for physics simulation
    dt: f64,
}

impl std::fmt::Debug for NbodyEntropy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NbodyEntropy")
            .field("particle_count", &self.particle_count)
            .field("counter", &self.counter)
            .finish()
    }
}

impl Default for NbodyEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl NbodyEntropy {
    /// Create a new system, seeded from system time
    pub fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        Self::from_seed(seed.to_le_bytes())
    }

    /// Create with custom particle count
    pub fn with_particle_count(particle_count: usize) -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let mut rng = Self::from_seed(seed.to_le_bytes());
        rng.set_particle_count(particle_count);
        rng
    }

    /// Set the number of active particles
    pub fn set_particle_count(&mut self, count: usize) {
        self.particle_count = count.min(MAX_PARTICLE_COUNT).max(ATTRACTOR_COUNT + 4);
    }

    /// Get current particle count
    pub fn particle_count(&self) -> usize {
        self.particle_count
    }

    /// Set connection threshold for graph-based queries
    pub fn set_connection_threshold(&mut self, threshold: f64) {
        self.connection_threshold = threshold.clamp(0.01, 1.0);
    }

    /// Step the n-body simulation forward
    /// Attractors interact with each other; followers are influenced by attractors
    #[inline]
    fn step_physics(&mut self) {
        let dt = self.dt;

        // Compute forces on attractors from other attractors (O(m²) where m = ATTRACTOR_COUNT)
        let mut attractor_forces = [[0.0f64; 2]; ATTRACTOR_COUNT];

        for i in 0..ATTRACTOR_COUNT {
            for j in (i + 1)..ATTRACTOR_COUNT {
                let pi = self.particles[i].pos;
                let pj = self.particles[j].pos;

                // Handle wrapping - use shortest distance
                let mut dx = pj[0] - pi[0];
                let mut dy = pj[1] - pi[1];

                // Wrap to nearest image
                if dx > BOUNDARY / 2.0 {
                    dx -= BOUNDARY;
                }
                if dx < -BOUNDARY / 2.0 {
                    dx += BOUNDARY;
                }
                if dy > BOUNDARY / 2.0 {
                    dy -= BOUNDARY;
                }
                if dy < -BOUNDARY / 2.0 {
                    dy += BOUNDARY;
                }

                let dist_sq = dx * dx + dy * dy + SOFTENING;
                let dist = dist_sq.sqrt();
                let force = G * self.particles[i].mass * self.particles[j].mass / dist_sq;

                let fx = force * dx / dist;
                let fy = force * dy / dist;

                attractor_forces[i][0] += fx;
                attractor_forces[i][1] += fy;
                attractor_forces[j][0] -= fx;
                attractor_forces[j][1] -= fy;
            }
        }

        // Update attractor velocities and positions
        for i in 0..ATTRACTOR_COUNT {
            let p = &mut self.particles[i];
            p.vel[0] += attractor_forces[i][0] * dt / p.mass;
            p.vel[1] += attractor_forces[i][1] * dt / p.mass;
            p.vel[0] *= DAMPING;
            p.vel[1] *= DAMPING;
            p.pos[0] = wrap(p.pos[0] + p.vel[0] * dt);
            p.pos[1] = wrap(p.pos[1] + p.vel[1] * dt);
        }

        // Update followers - influenced by attractors (O(m*n))
        for i in ATTRACTOR_COUNT..self.particle_count {
            let mut fx = 0.0;
            let mut fy = 0.0;

            // Sample a few attractors for speed (not all)
            let sample_count = 4.min(ATTRACTOR_COUNT);
            let step = ATTRACTOR_COUNT / sample_count;
            for s in 0..sample_count {
                let j = s * step;
                let pi = self.particles[i].pos;
                let pj = self.particles[j].pos;

                let mut dx = pj[0] - pi[0];
                let mut dy = pj[1] - pi[1];

                if dx > BOUNDARY / 2.0 {
                    dx -= BOUNDARY;
                }
                if dx < -BOUNDARY / 2.0 {
                    dx += BOUNDARY;
                }
                if dy > BOUNDARY / 2.0 {
                    dy -= BOUNDARY;
                }
                if dy < -BOUNDARY / 2.0 {
                    dy += BOUNDARY;
                }

                let dist_sq = dx * dx + dy * dy + SOFTENING;
                let dist = dist_sq.sqrt();
                let force = G * self.particles[j].mass / dist_sq;

                fx += force * dx / dist;
                fy += force * dy / dist;
            }

            let p = &mut self.particles[i];
            p.vel[0] += fx * dt;
            p.vel[1] += fy * dt;
            p.vel[0] *= DAMPING;
            p.vel[1] *= DAMPING;
            p.pos[0] = wrap(p.pos[0] + p.vel[0] * dt);
            p.pos[1] = wrap(p.pos[1] + p.vel[1] * dt);
        }
    }

    /// Query the distance between two particles
    #[inline(always)]
    fn query_distance(&self, i: usize, j: usize) -> f64 {
        let pi = self.particles[i % self.particle_count].position();
        let pj = self.particles[j % self.particle_count].position();
        let dx = pi[0] - pj[0];
        let dy = pi[1] - pj[1];
        (dx * dx + dy * dy).sqrt()
    }

    /// Query the angle formed by three particles (at vertex j)
    #[inline]
    fn query_angle(&self, i: usize, j: usize, k: usize) -> f64 {
        let pi = self.particles[i % self.particle_count].position();
        let pj = self.particles[j % self.particle_count].position();
        let pk = self.particles[k % self.particle_count].position();

        let v1 = [pi[0] - pj[0], pi[1] - pj[1]];
        let v2 = [pk[0] - pj[0], pk[1] - pj[1]];

        let dot = v1[0] * v2[0] + v1[1] * v2[1];
        let mag_sq1 = v1[0] * v1[0] + v1[1] * v1[1];
        let mag_sq2 = v2[0] * v2[0] + v2[1] * v2[1];

        if mag_sq1 < 1e-20 || mag_sq2 < 1e-20 {
            return 0.0;
        }

        let cos_angle = dot / (mag_sq1 * mag_sq2).sqrt();
        fast_acos(cos_angle)
    }

    /// Query velocity magnitude
    #[inline]
    fn query_velocity(&self, i: usize) -> f64 {
        let p = &self.particles[i % self.particle_count];
        (p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1]).sqrt()
    }

    /// Query how many particles are within threshold distance of particle i
    #[inline]
    fn query_neighbors(&self, i: usize, threshold: f64) -> usize {
        let mut count = 0;
        let sample_count = 24.min(self.particle_count);
        let step = self.particle_count / sample_count;
        for s in 0..sample_count {
            let j = (i + s * step + 1) % self.particle_count;
            if self.query_distance(i, j) < threshold {
                count += 1;
            }
        }
        count * self.particle_count / sample_count
    }

    /// Query connectivity
    #[inline]
    fn query_connectivity(&self, i: usize, j: usize) -> bool {
        let i = i % self.particle_count;
        let j = j % self.particle_count;

        if i == j {
            return true;
        }

        if self.query_distance(i, j) < self.connection_threshold {
            return true;
        }

        let sample_count = 6.min(self.particle_count);
        for step in 0..sample_count {
            let k = (i + step * 31) % self.particle_count;
            if k != i
                && k != j
                && self.query_distance(i, k) < self.connection_threshold
                && self.query_distance(k, j) < self.connection_threshold
            {
                return true;
            }
        }

        false
    }

    /// Mix data into the internal state
    #[inline]
    fn mix(&mut self, data: &[u8]) {
        if self.counter % 128 != 0 {
            let data_u64 = if data.len() >= 8 {
                u64::from_le_bytes(data[0..8].try_into().unwrap())
            } else {
                let mut buf = [0u8; 8];
                buf[..data.len()].copy_from_slice(data);
                u64::from_le_bytes(buf)
            };

            let mut state_u64 = u64::from_le_bytes(self.state[0..8].try_into().unwrap());
            state_u64 = state_u64.wrapping_add(data_u64);
            state_u64 ^= state_u64 << 13;
            state_u64 ^= state_u64 >> 7;
            state_u64 ^= state_u64 << 17;
            state_u64 = state_u64.wrapping_mul(0x2545F4914F6CDD1D);
            self.state[0..8].copy_from_slice(&state_u64.to_le_bytes());

            let mut state_u64_2 = u64::from_le_bytes(self.state[8..16].try_into().unwrap());
            state_u64_2 ^= state_u64.rotate_left(23);
            state_u64_2 = state_u64_2.wrapping_mul(0x9E3779B97F4A7C15);
            self.state[8..16].copy_from_slice(&state_u64_2.to_le_bytes());

            self.state.rotate_left(5);
        } else {
            let mut hasher = blake3::Hasher::new_keyed(&self.hasher_key);
            hasher.update(&self.state);
            hasher.update(data);
            hasher.update(&self.counter.to_le_bytes());
            self.state = *hasher.finalize().as_bytes();
        }

        if self.counter % 8192 == 0 {
            let mut key_hasher = blake3::Hasher::new();
            key_hasher.update(&self.state);
            key_hasher.update(b"key_update");
            self.hasher_key = *key_hasher.finalize().as_bytes();
        }
    }

    /// Perturb a particle based on feedback
    #[inline(always)]
    fn perturb(&mut self, i: usize, amount: f64) {
        let p = &mut self.particles[i % self.particle_count];
        // Add small velocity perturbation
        p.vel[0] += (amount - 0.5) * 0.001;
        p.vel[1] += (amount - 0.5) * 0.0007;
    }

    /// Execute a query and return the result as bytes
    #[inline]
    fn execute_query(&self, query_type: QueryType, i: usize, j: usize, k: usize) -> [u8; 8] {
        let result = match query_type {
            QueryType::Distance => {
                let dist = self.query_distance(i, j);
                (dist * 1e15) as u64
            }
            QueryType::Angle => {
                let angle = self.query_angle(i, j, k);
                (angle * 1e15) as u64
            }
            QueryType::Neighbors => {
                let threshold = 0.1 + (self.state[0] as f64 / 255.0) * 0.3;
                self.query_neighbors(i, threshold) as u64
            }
            QueryType::CombinedDistance => {
                let d1 = self.query_distance(i, j);
                let d2 = self.query_distance(j, k);
                ((d1 * d2) * 1e15) as u64
            }
            QueryType::Connectivity => {
                if self.query_connectivity(i, j) {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            }
            QueryType::Velocity => {
                let v = self.query_velocity(i);
                (v * 1e18) as u64
            }
        };

        result.to_le_bytes()
    }

    /// Generate the next random u64 using the feedback loop
    #[inline]
    fn generate(&mut self) -> u64 {
        self.counter = self.counter.wrapping_add(1);

        // Step physics simulation (chaotic n-body dynamics)
        self.step_physics();

        // Use current state to determine query parameters
        let state_u64 = u64::from_le_bytes(self.state[0..8].try_into().unwrap());
        let query_type = QueryType::from_index(state_u64);
        let i = ((state_u64 >> 3) & 0x3FF) as usize;
        let j = ((state_u64 >> 13) & 0x3FF) as usize;
        let k = ((state_u64 >> 23) & 0x3FF) as usize;

        // Execute query
        let query_result = self.execute_query(query_type, i, j, k);

        // Mix query result into state
        self.mix(&query_result);

        // Perturb particles based on result (feedback!)
        let perturb_amount =
            u16::from_le_bytes([query_result[0], query_result[1]]) as f64 / 65535.0;
        self.perturb(i, perturb_amount);
        self.perturb(j, perturb_amount * 0.5);

        // Vary time step slightly based on state
        self.dt = 0.01 + (self.state[8] as f64 / 255.0) * 0.005;

        // Return mixed state as output
        u64::from_le_bytes(self.state[0..8].try_into().unwrap())
    }
}

impl SeedableRng for NbodyEntropy {
    type Seed = [u8; 8];

    fn from_seed(seed: Self::Seed) -> Self {
        // Expand seed using BLAKE3
        let mut hasher = blake3::Hasher::new();
        hasher.update(&seed);
        hasher.update(b"chaos_entropy_init_v2");
        let initial_state = *hasher.finalize().as_bytes();

        // Generate hasher key
        let mut key_hasher = blake3::Hasher::new();
        key_hasher.update(&seed);
        key_hasher.update(b"hasher_key_v2");
        let hasher_key = *key_hasher.finalize().as_bytes();

        // Use expanded state to seed particle generation
        let mut state = u64::from_le_bytes(initial_state[0..8].try_into().unwrap());
        let mut particles = Vec::with_capacity(MAX_PARTICLE_COUNT);

        for i in 0..MAX_PARTICLE_COUNT {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            let pos = [
                (state & 0xFFFF) as f64 / 65535.0,
                ((state >> 16) & 0xFFFF) as f64 / 65535.0,
            ];

            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            // Small random initial velocities
            let vel = [
                ((state & 0xFFFF) as f64 / 65535.0 - 0.5) * 0.01,
                (((state >> 16) & 0xFFFF) as f64 / 65535.0 - 0.5) * 0.01,
            ];

            let is_attractor = i < ATTRACTOR_COUNT;
            let mass = if is_attractor { 10.0 } else { 1.0 };

            particles.push(Particle {
                pos,
                vel,
                mass,
                is_attractor,
            });
        }

        Self {
            particles,
            particle_count: DEFAULT_PARTICLE_COUNT,
            state: initial_state,
            counter: 0,
            connection_threshold: 0.15,
            hasher_key,
            dt: 0.01,
        }
    }
}

impl RngCore for NbodyEntropy {
    fn next_u32(&mut self) -> u32 {
        self.generate() as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.generate()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut i = 0;
        while i < dest.len() {
            let val = self.generate();
            let bytes = val.to_le_bytes();
            let remaining = dest.len() - i;
            let to_copy = remaining.min(8);
            dest[i..i + to_copy].copy_from_slice(&bytes[..to_copy]);
            i += to_copy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic() {
        let mut rng1 = NbodyEntropy::from_seed([1, 2, 3, 4, 5, 6, 7, 8]);
        let mut rng2 = NbodyEntropy::from_seed([1, 2, 3, 4, 5, 6, 7, 8]);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_different_seeds() {
        let mut rng1 = NbodyEntropy::from_seed([1, 2, 3, 4, 5, 6, 7, 8]);
        let mut rng2 = NbodyEntropy::from_seed([8, 7, 6, 5, 4, 3, 2, 1]);

        let mut same = 0;
        for _ in 0..100 {
            if rng1.next_u64() == rng2.next_u64() {
                same += 1;
            }
        }
        assert!(same < 5, "Too many collisions between different seeds");
    }

    #[test]
    fn test_fill_bytes() {
        let mut rng = NbodyEntropy::new();
        let mut buf1 = [0u8; 32];
        let mut buf2 = [0u8; 32];

        rng.fill_bytes(&mut buf1);
        rng.fill_bytes(&mut buf2);

        assert_ne!(buf1, buf2);
    }

    #[test]
    fn test_bit_distribution() {
        let mut rng = NbodyEntropy::from_seed([42, 0, 0, 0, 0, 0, 0, 0]);
        let samples = 10000;

        let mut ones = 0u64;
        for _ in 0..samples {
            ones += rng.next_u64().count_ones() as u64;
        }

        let expected = samples as u64 * 32;
        let deviation = (ones as f64 - expected as f64).abs() / expected as f64;
        assert!(
            deviation < 0.02,
            "Bit distribution deviation too high: {}",
            deviation
        );
    }

    #[test]
    fn test_attractor_count() {
        let rng = NbodyEntropy::new();
        let attractor_count = rng.particles.iter().filter(|p| p.is_attractor).count();
        assert_eq!(attractor_count, ATTRACTOR_COUNT);
    }
}
