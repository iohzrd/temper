# Temper

GPU-accelerated Langevin dynamics for optimization, sampling, and entropy generation.

The architecture uses particle dynamics as a unified computational medium. The same update equation produces optimization, sampling, or entropy depending solely on temperature.

## Background

Temperature controls the exploration-exploitation tradeoff in stochastic systems. This is established statistical mechanics (Kirkpatrick 1983, Liu & Wang 2016):

| Temperature | Mode     | Behavior                                   |
| ----------- | -------- | ------------------------------------------ |
| T → 0       | Optimize | Gradient descent to minima                 |
| T ~ 0.1     | Sample   | Boltzmann distribution p(x) ∝ exp(-E(x)/T) |
| T >> 1      | Entropy  | Chaotic exploration → random bits          |

The update equation combines overdamped Langevin dynamics with SVGD repulsion:

```
dx = -γ∇E(x)·dt + F_svgd·dt + √(2γT)·dW
```

The SVGD term (Liu & Wang 2016) adds inter-particle repulsion to prevent mode collapse.

## Quick Start

```bash
# Visualizations (require viz feature)
cargo run --release --features "gpu viz" --bin landscape-viz
cargo run --release --features "gpu viz" --bin parallel-tempering-viz

# Benchmarks
cargo run --release --features gpu --bin benchmark
cargo run --release --features gpu --bin optimizer-comparison

# Demos
cargo run --release --features gpu --bin mode-discovery        # Find all modes
cargo run --release --features gpu --bin bayesian-uncertainty  # Uncertainty quantification
```

See [bin/README.md](src/bin/README.md) for full list.

## Library Usage

```rust
use temper::{ThermodynamicSystem, LossFunction};

let mut system = ThermodynamicSystem::with_loss_function(
    1000,                    // particles
    4,                       // dimensions
    0.1,                     // temperature
    LossFunction::Rastrigin
);

// Simulated annealing
for step in 0..5000 {
    let t = 1.0 * (0.001_f32).powf(step as f32 / 5000.0);
    system.set_temperature(t);
    system.step();
}

let particles = system.read_particles();
let best = particles.iter()
    .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
    .unwrap();
println!("Best: {}", best.energy);
```

### Custom Loss Functions

```rust
use temper::expr::*;

// Griewank function
let griewank = const_(1.0)
    + sum_dims(|x, _| x.powi(2) / 4000.0)
    - prod_dims(|x, i| cos(x / sqrt(i + 1.0)));

let mut system = ThermodynamicSystem::with_expr(500, 4, 1.0, griewank);
```

### Performance Tuning

```rust
system.set_repulsion_samples(0);   // Skip repulsion (faster, optimization only)
system.set_repulsion_samples(64);  // Sample 64 particles (default)
```

## Features

- **Optimization**: Simulated annealing, adaptive cooling with reheat, parallel tempering
- **Sampling**: SVGD-based posterior sampling, uncertainty quantification
- **Entropy**: High-temperature mode for randomness extraction, implements `RngCore`
- **GPU**: wgpu-based, O(nK) repulsion subsampling
- **Expression DSL**: Custom loss functions compile to WGSL

### Built-in Loss Functions

Sphere, Rosenbrock, Rastrigin, Ackley, Schwefel, MLP-XOR (9 params), MLP-Deep (37 params)

## Architecture

```
src/
├── thermodynamic/     # Core particle system
│   ├── system.rs      # GPU implementation
│   ├── scheduler.rs   # Adaptive annealing
│   └── rng.rs         # RngCore implementation
├── expr/              # Expression DSL
├── viz/               # Visualization utilities
└── bin/               # Demos and benchmarks
```

## Connection to Diffusion Models

Diffusion models use related math: for a Boltzmann distribution p(x) ∝ exp(-E(x)/T), the score function is ∇log p(x) = -∇E(x)/T. Reverse diffusion can be viewed as temperature annealing with a learned energy function.

## References

1. Kirkpatrick et al. (1983). Optimization by simulated annealing. _Science_
2. Liu & Wang (2016). Stein variational gradient descent. _NeurIPS_
3. Swendsen & Wang (1986). Replica Monte Carlo simulation. _PRL_

## License

MIT
