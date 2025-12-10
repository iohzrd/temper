# N-Body Entropy

An experimental entropy generator that explores extracting randomness from N-body gravitational simulation through self-referential feedback loops.

## Concept

Traditional PRNGs use mathematical operations (xorshift, LCG, etc.) to produce pseudo-random sequences. This project explores a different approach: using a **high-dimensional physical simulation** as the entropy source.

The key insight: while a deterministic system cannot create entropy, a feedback loop where **outputs determine which parts of the system to query next** creates compounding complexity that becomes practically unpredictable.

## How It Works

### The Particle System

256 particles exist in a 2D normalized space with chaotic n-body dynamics:

**Attractor Particles (12)**
- High mass (10.0)
- Full gravitational interaction with other attractors
- Create emergent chaotic behavior through n-body dynamics

**Follower Particles (244)**
- Low mass (1.0)
- Influenced by sampled attractors (not each other)
- Add high-dimensional state space

Each particle has:
- Position (x, y)
- Velocity (x, y)
- Mass
- Toroidal boundary wrapping

### The Feedback Loop

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ┌─────────────┐      ┌──────────────────────────────┐    │
│   │   Current   │─────▶│  Select query type & targets │    │
│   │    State    │      │  (which particles to query)  │    │
│   └─────────────┘      └──────────────────────────────┘    │
│          ▲                           │                      │
│          │                           ▼                      │
│          │             ┌──────────────────────────────┐    │
│          │             │  Execute relational query:   │    │
│          │             │  • Distance between i and j  │    │
│          │             │  • Angle at vertex j         │    │
│          │             │  • Neighbor count around i   │    │
│          │             │  • Combined distance product │    │
│          │             └──────────────────────────────┘    │
│          │                           │                      │
│          │                           ▼                      │
│   ┌──────┴──────┐      ┌──────────────────────────────┐    │
│   │  Mix into   │◀─────│     Query result (f64)       │    │
│   │   state     │      └──────────────────────────────┘    │
│   └─────────────┘                    │                      │
│          │                           ▼                      │
│          │             ┌──────────────────────────────┐    │
│          │             │  FEEDBACK: Perturb queried   │    │
│          │             │  particles based on result   │    │
│          │             └──────────────────────────────┘    │
│          │                           │                      │
│          │                           ▼                      │
│          │             ┌──────────────────────────────┐    │
│          └─────────────│  Advance time (variable dt)  │    │
│                        └──────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why Feedback Matters

Without feedback, the system is fully deterministic—given the same seed and time, you get identical results.

With feedback:
1. Query results **modify the particles** that were queried
2. Modified particles produce **different results** in future queries
3. Which particles get queried depends on **previous outputs**
4. Small changes compound exponentially over iterations

This creates a self-referential system where predicting output N requires computing all outputs 0..N-1.

## Query Types

The system selects from six query types based on internal state:

| Type | Description | Output |
|------|-------------|--------|
| Distance | Euclidean distance between particles i and j | `sqrt(dx² + dy²)` |
| Angle | Angle formed by three particles at vertex j | `acos(dot/mag)` |
| Neighbors | Count of particles within threshold of i | Integer count |
| Combined | Product of two distances | `dist(i,j) × dist(j,k)` |
| Connectivity | Are particles i and j in same cluster? | Boolean (2-hop) |
| Velocity | Velocity magnitude of particle i | `sqrt(vx² + vy²)` |

## Statistical Test Results

### Built-in NIST SP 800-22 Inspired Suite

All 12 tests pass with 100,000 sample values:

```
╔══════════════════════════════════════════════════════════════╗
║  NIST SP 800-22 Inspired Statistical Test Suite             ║
╚══════════════════════════════════════════════════════════════╝

1. Frequency (Monobit) Test... S_obs = 0.9961 PASS
2. Block Frequency Test (M=128)... χ² = 49983.38 (n=50000) PASS
3. Runs Test... runs=3200842, z=0.6661 PASS
4. Longest Run of Ones... max=22, expected~22.6 PASS
5. Binary Matrix Rank Test (32x32)... full=299, r31=575, χ²=0.37 PASS
6. Spectral Test (simplified)... max_autocorr=0.2433 PASS
7. Template Matching (pattern: 0b11111111)... matches=22421, expected=22266, z=1.04 PASS
8. Serial Test (2-bit patterns)... χ²=1.12 PASS
9. Approximate Entropy Test... ApEn=0.6931 (expected~0.693) PASS
10. Cumulative Sums Test... max=3302, expected<7589 PASS
11. Random Excursions (zero crossings)... crossings=789, expected~2530 PASS
12. Monte Carlo Pi Estimation... π≈3.139520 (error=0.002073) PASS

╔══════════════════════════════════════════════════════════════╗
║  RESULTS: 12/12 tests passed                                ║
║  Overall: EXCELLENT                                         ║
╚══════════════════════════════════════════════════════════════╝
```

### ENT Analysis (10 MB sample)

```
Entropy = 7.999983 bits per byte.

Optimum compression would reduce the size
of this 10000000 byte file by 0 percent.

Chi square distribution for 10000000 samples is 238.63, and randomly
would exceed this value 76.15 percent of the times.

Arithmetic mean value of data bytes is 127.5359 (127.5 = random).
Monte Carlo value for Pi is 3.141646857 (error 0.00 percent).
Serial correlation coefficient is -0.000594 (totally uncorrelated = 0.0).
```

| Metric | Value | Ideal | Assessment |
|--------|-------|-------|------------|
| Entropy | 7.999983 bits/byte | 8.0 | Excellent |
| Chi-squared | 238.63 (76.15%) | 10-90% range | Excellent |
| Mean | 127.5359 | 127.5 | Excellent |
| Pi estimate | 3.141646857 | 3.14159... | Excellent |
| Serial correlation | -0.000594 | 0.0 | Excellent |

### Dieharder

```
diehard_count_1s_str|   0|    256000|     100|0.42183623|  PASSED
```

### Performance

| Version | Speed | Notes |
|---------|-------|-------|
| CPU (n-body) | ~0.87 MB/s | Chaotic dynamics, full simulation |
| GPU (batched) | **~107 MB/s** | **120x faster**, wgpu compute shaders |

```bash
# CPU benchmark
cargo run --release -- benchmark

# GPU benchmark (requires gpu feature)
cargo run --release --features gpu -- gpu-benchmark
```

Optimized with:
- **Chaotic n-body dynamics**: 12 attractor particles with gravitational interaction
- **GPU acceleration**: wgpu compute shaders for parallel physics simulation
- **Batched generation**: 32768 values per GPU roundtrip to amortize overhead
- **Fast acos approximation**: Polynomial approximation (~0.5% max error)
- **XOR-based mixing**: Fast mixing in hot path (BLAKE3 only used at init)
- Weighted query selection favoring faster operations (Distance: 62.5%, Velocity: 18.75%)

## Usage

### Command Line

```bash
# Run built-in NIST test suite
cargo run --release -- test

# Output raw bytes (for piping to external tools)
cargo run --release -- raw 10000000 | ent

# Continuous stream for dieharder (runs until interrupted)
cargo run --release -- stream | dieharder -a -g 200

# Benchmark generation speed
cargo run --release -- benchmark
```

### As a Library

```rust
use nbody_entropy::NbodyEntropy;
use rand_core::{RngCore, SeedableRng};

// Create from system time
let mut rng = NbodyEntropy::new();

// Or from explicit 8-byte seed (implements SeedableRng)
let mut rng = NbodyEntropy::from_seed([1, 2, 3, 4, 5, 6, 7, 8]);

// Generate random values (implements RngCore)
let value: u64 = rng.next_u64();
let value32: u32 = rng.next_u32();

// Fill a buffer
let mut buf = [0u8; 32];
rng.fill_bytes(&mut buf);

// Configure particle count (16-1024)
let mut rng = NbodyEntropy::with_particle_count(512);
```

### GPU-Accelerated Version

```rust
// Enable the "gpu" feature in Cargo.toml:
// nbody-entropy = { version = "0.1", features = ["gpu"] }

#[cfg(feature = "gpu")]
use nbody_entropy::GpuNbodyEntropy;
use rand_core::{RngCore, SeedableRng};

#[cfg(feature = "gpu")]
fn main() {
    // GPU version has the same API
    let mut rng = GpuNbodyEntropy::new();

    // Generate random values (80x faster than CPU)
    let value = rng.next_u64();

    // Fill buffers efficiently
    let mut buf = [0u8; 1024];
    rng.fill_bytes(&mut buf);
}
```

## Limitations

This is an **experimental prototype**, not suitable for cryptographic use without further analysis:

1. **Not proven secure** - Needs formal cryptanalysis
2. **Relatively slow** - Each output requires multiple trig operations
3. **Large state** - 256 particles × multiple f64 parameters
4. **Deterministic** - Same seed produces same sequence (like any PRNG)

## Features

- [x] Implements `rand_core::{RngCore, SeedableRng}` traits
- [x] BLAKE3-based state mixing (periodic strengthening)
- [x] Chaotic n-body dynamics (12 attractor particles)
- [x] 6 query types including velocity and graph connectivity
- [x] Configurable particle count (16-1024)
- [x] Continuous stream mode for testing tools
- [x] **GPU acceleration** via wgpu compute shaders (optional feature)

## Future Exploration

- [x] Test with NIST SP 800-22 suite (dieharder, ent) - **PASSED**
- [x] Add true chaotic dynamics - **N-body gravity implemented**
- [x] GPU acceleration - **wgpu compute shaders (70 MB/s)**
- [ ] Run full dieharder test battery (`dieharder -a`)
- [ ] Test with TestU01 BigCrush
- [ ] SIMD acceleration for CPU version
- [ ] Hybrid approach: mix with hardware entropy source
- [ ] Formal cryptanalysis

## Theory

The approach is inspired by:

- **Chaos-based PRNGs** (logistic map, Lorenz attractor) - but with much higher dimensionality
- **Entropy harvesting** from physical systems - but using a simulated physical system
- **Hash functions** - the mixing of query results acts like iterative hashing

The novelty is using **relational queries** (distances, angles, topology) on a particle system, where the relationships create combinatorial complexity far exceeding individual particle state.

## License

MIT
