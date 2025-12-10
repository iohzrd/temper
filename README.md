# N-Body Entropy

An experimental entropy generator that extracts randomness from GPU-accelerated N-body gravitational simulation with chaotic dynamics.

![N-Body Visualization](image.png)

## Concept

Traditional PRNGs use mathematical operations (xorshift, LCG, etc.) to produce pseudo-random sequences. This project explores a different approach: using a **chaotic physical simulation** as the entropy source.

The N-body gravitational problem is inherently chaotic - small differences in initial conditions lead to exponentially diverging trajectories. Combined with a slingshot mechanic that adds tangential velocity boosts on close approach, the system maintains continuous chaotic motion without collapsing into stable configurations.

## How It Works

### The Particle System

64 particles exist in a 2D toroidal space with chaotic n-body dynamics:

**Attractor Particles (8)**

- High mass (10.0)
- Full gravitational interaction with all other attractors
- Create emergent chaotic behavior through n-body dynamics
- Slingshot effect on close approach prevents clumping

**Follower Particles (56)**

- Low mass (1.0)
- Influenced by sampled attractors (not each other)
- Add high-dimensional state space

### Physics

Each particle experiences:

- **Gravitational attraction**: Standard inverse-square law between particles
- **Slingshot effect**: Tangential velocity boost on close approach (radius < 0.08), creating orbital dynamics
- **Velocity damping**: Slight damping (0.999) for stability
- **Toroidal wrapping**: Particles wrap around boundaries

The slingshot mechanic is key - without it, particles eventually clump into a single mass (bad for entropy). With it, particles continuously orbit and interact chaotically.

### Entropy Generation

1. Seed from OS entropy (`/dev/urandom` on Linux)
2. Run N-body physics simulation
3. Mix particle positions/velocities with xorshift operations
4. Extract random bytes from mixed state
5. Repeat

This acts as an **entropy amplifier** - a small amount of true randomness (8 bytes) is expanded into a high-throughput stream (~5 GB/s) through chaotic dynamics.

## Performance

| Version | Speed       | Notes                                    |
| ------- | ----------- | ---------------------------------------- |
| CPU     | ~0.87 MB/s  | Full n-body simulation                   |
| GPU     | **~5 GB/s** | wgpu compute shaders, batched generation |

```bash
# GPU benchmark
cargo run --release --features gpu --bin nbody-entropy -- gpu-benchmark
```

## Usage

### Command Line

```bash
# Run built-in NIST test suite
cargo run --release --bin nbody-entropy -- test

# GPU test suite (much faster)
cargo run --release --features gpu --bin nbody-entropy -- gpu-test

# Output raw bytes for external tools
cargo run --release --features gpu --bin nbody-entropy -- gpu-raw 10000000 | ent

# Continuous stream for dieharder
cargo run --release --features gpu --bin nbody-entropy -- gpu-stream | dieharder -a -g 200

# Benchmark
cargo run --release --features gpu --bin nbody-entropy -- gpu-benchmark
```

### Visualization

Watch the particles dance:

```bash
cargo run --release --features viz --bin nbody-viz
```

- Orange particles: Attractors (8)
- Blue particles: Followers (56)
- Lines show proximity relationships

### As a Library

```rust
use nbody_entropy::NbodyEntropy;
use rand_core::{RngCore, SeedableRng};

// Create from OS entropy (/dev/urandom on Linux)
let mut rng = NbodyEntropy::new();

// Or from explicit seed (for reproducible sequences)
let mut rng = NbodyEntropy::from_seed([1, 2, 3, 4, 5, 6, 7, 8]);

// Generate random values (implements RngCore)
let value: u64 = rng.next_u64();

// Fill a buffer
let mut buf = [0u8; 32];
rng.fill_bytes(&mut buf);
```

### GPU-Accelerated Version

```rust
#[cfg(feature = "gpu")]
use nbody_entropy::GpuNbodyEntropy;
use rand_core::RngCore;

// GPU version - seeded from OS entropy, ~5000x faster
let mut rng = GpuNbodyEntropy::new();
let value = rng.next_u64();
```

## Features

```toml
[dependencies]
nbody-entropy = "0.1"

# With GPU acceleration
nbody-entropy = { version = "0.1", features = ["gpu"] }

# With visualization
nbody-entropy = { version = "0.1", features = ["viz"] }
```

## Statistical Testing

Passes dieharder tests at ~5 GB/s throughput:

```bash
cargo run --release --features gpu --bin nbody-entropy -- gpu-stream | dieharder -a -g 200
```

## Limitations

This is an **experimental project**:

1. **Not cryptographically proven** - Needs formal analysis
2. **Non-deterministic on GPU** - Parallel floating point operations may vary between runs (actually a feature for entropy)
3. **Requires GPU** - For practical speeds

## How It Differs From Other Chaos PRNGs

Most chaos-based PRNGs use simple systems (logistic map, Lorenz attractor). This uses:

- **High dimensionality**: 64 particles × 4 state variables = 256 dimensions
- **N-body interactions**: O(n²) gravitational relationships
- **Slingshot dynamics**: Prevents collapse into attracting fixed points
- **GPU parallelism**: Each particle computed in parallel

## License

MIT
