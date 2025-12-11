# N-Body Entropy

A GPU-accelerated **unified thermodynamic particle system** that demonstrates optimization, Bayesian sampling, and entropy generation are all the same algorithm at different temperatures.

![N-Body Visualization](image.png)

## Core Thesis

Traditional approaches treat optimization, MCMC sampling, and random number generation as separate problems. This project shows they're all **points on a temperature continuum**:

| Temperature | Mode | Behavior |
|-------------|------|----------|
| T → 0 | **Optimize** | Particles converge to loss minima (simulated annealing) |
| T ~ 0.1 | **Sample** | Particles sample from Boltzmann distribution p(x) ∝ exp(-E(x)/T) |
| T >> 1 | **Entropy** | Chaotic exploration extracts random bits |

The unified update equation:
```
dx = -γ∇E(x)·dt + repulsion·dt + √(2γT·dt)·dW
```

## Features

### Neural Network Training
Train real MLPs using particle-based optimization:
- **XOR classification**: 9-parameter MLP achieves 100% accuracy
- **Circles classification**: 37-parameter deep MLP (2→4→4→1) achieves 100% accuracy
- **Spiral classification**: Challenging non-linear separation

### Optimization Benchmarks
Built-in loss functions for testing:
- Sphere (convex baseline)
- Rosenbrock (narrow valley)
- Rastrigin (highly multimodal)
- Ackley (flat outer region)

### Advanced Techniques
- **Parallel Tempering**: Replica exchange between temperature levels for better global search
- **Bayesian Posterior Sampling**: Verified Boltzmann distribution sampling at moderate temperatures
- **Simulated Annealing**: Automatic temperature scheduling

### GPU Performance
Optimized for large particle counts:
- **O(nK) subsampling**: Reduces O(n²) repulsion to O(nK) with configurable K
- **21x speedup** with K=0 (skip repulsion for pure optimization)
- **6-7x speedup** with K=64 (default for SVGD sampling)
- **16k particles** supported at 72+ steps/sec

## Quick Start

```bash
# Run the GPU performance benchmark
cargo run --release --features gpu --bin benchmark

# Train a deep neural network on circles
cargo run --release --features gpu --bin deep-nn-benchmark

# Compare optimizers (Thermodynamic vs SGD vs Adam)
cargo run --release --features gpu --bin optimizer-comparison

# Parallel tempering demo
cargo run --release --features gpu --bin parallel-tempering

# Bayesian sampling visualization
cargo run --release --features gpu --bin bayesian-sampling

# Hyperparameter tuning demo
cargo run --release --features gpu --bin hyperparam-tuning

# Interactive visualization
cargo run --release --features "gpu viz" --bin thermodynamic-viz
```

## Usage as Library

```rust
use nbody_entropy::thermodynamic::{ThermodynamicSystem, LossFunction};

// Create system: 1000 particles, 4 dimensions, temperature 0.1
let mut system = ThermodynamicSystem::with_loss_function(
    1000,                    // particles
    4,                       // dimensions
    0.1,                     // temperature (sampling mode)
    LossFunction::Rastrigin  // loss function
);

// Run simulation
for _ in 0..1000 {
    system.step();
}

// Read results
let particles = system.read_particles();
let best = particles.iter()
    .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
    .unwrap();
println!("Best loss: {}", best.energy);
```

### Simulated Annealing

```rust
// Annealing: start hot, cool down
for step in 0..5000 {
    let progress = step as f32 / 5000.0;
    let temp = 1.0 * (0.0001_f32 / 1.0).powf(progress);
    system.set_temperature(temp);
    system.step();
}
```

### Performance Tuning

```rust
// For pure optimization (fastest)
system.set_repulsion_samples(0);  // Skip repulsion entirely

// For SVGD sampling (default)
system.set_repulsion_samples(64); // Sample 64 particles

// For maximum accuracy (slowest)
system.set_repulsion_samples(particle_count as u32); // Full O(n²)
```

## Available Loss Functions

| Loss | Enum | Dim | Description |
|------|------|-----|-------------|
| Neural Net 2D | `NeuralNet2D` | 2 | Simple 2-param network |
| Multimodal | `Multimodal` | N | 2^(N/2) global minima |
| Rosenbrock | `Rosenbrock` | N | Banana valley, min at (1,...,1) |
| Rastrigin | `Rastrigin` | N | Highly multimodal, min at origin |
| Ackley | `Ackley` | N | Flat outer region with central hole |
| Sphere | `Sphere` | N | Simple convex, min at origin |
| MLP XOR | `MlpXor` | 9 | Real MLP on XOR problem |
| MLP Spiral | `MlpSpiral` | 9 | Spiral classification |
| MLP Deep | `MlpDeep` | 37 | 3-layer MLP on circles |

## Benchmark Results

### GPU Performance (RTX-class GPU)

```
REPULSION SAMPLING COMPARISON (1000 particles, dim=4)
     Samples    Steps/sec      µs/step      Speedup
        skip      10827.2         92.4         21.1x
          64       3228.4        309.8          6.3x
        1000        482.2       2074.0          1.0x

SCALING WITH PARTICLE COUNT (K=64 samples)
  Particles  Steps/sec      µs/step
       1000     2910.0        343.6
       4000      499.3       2002.8
      10000      149.6       6686.2
      16000       71.7      13942.6
```

### Optimization Quality

| Test | Result |
|------|--------|
| Deep MLP (37 params) | 100% accuracy on circles |
| XOR MLP | 100% accuracy |
| Parallel Tempering vs Annealing | Wins on Rastrigin (3.15 vs 9.18) |

## Entropy Generation

At high temperature (T >> 1), the system generates cryptographic-quality randomness:

```bash
# Stream entropy to dieharder
cargo run --release --features gpu -- stream | dieharder -a -g 200

# Test with ent
cargo run --release --features gpu -- raw 10000000 | ent
```

## Architecture

```
src/
├── thermodynamic.rs      # Core unified system
├── shaders/
│   └── thermodynamic.wgsl # GPU compute shader
├── benchmark.rs          # Performance profiling
├── deep_nn_benchmark.rs  # 37-param MLP test
├── optimizer_comparison.rs # SGD/Adam/Thermodynamic
├── parallel_tempering.rs # Replica exchange
├── bayesian_sampling.rs  # Posterior sampling demo
└── hyperparam_tuning.rs  # ML hyperparameter search
```

## Dependencies

- `wgpu` - GPU compute via WebGPU
- `bytemuck` - Safe transmutation
- `pollster` - Async runtime for GPU

## License

MIT
