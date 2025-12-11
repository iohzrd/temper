# Unified Thermodynamic Particle Systems: A Computational Primitive

## Abstract

We demonstrate that a single GPU-accelerated interacting particle system can perform three fundamentally different computations—**optimization**, **Bayesian inference**, and **entropy generation**—controlled solely by a temperature parameter. This suggests that thermodynamic particle dynamics may serve as a universal computational primitive, with temperature selecting the type of computation performed.

## The Noise Paradox

Modern AI has a peculiar relationship with randomness:

1. **AI needs noise**: Dropout, stochastic gradient descent, diffusion models, MCMC sampling, exploration in RL—noise is _essential_ to modern machine learning.

2. **Hardware fights noise**: Traditional IC design spends enormous effort suppressing thermal noise, using error correction, and ensuring deterministic operation.

3. **Then we simulate it back**: After building perfectly deterministic hardware, we run PRNGs and add synthetic noise to our computations.

This is deeply inefficient. We're:

- Paying power costs to suppress physical noise
- Paying compute costs to generate synthetic noise
- Getting _worse_ randomness than physics provides for free

**Temper cuts through this paradox** by using physical dynamics directly. At high temperature, particle motion _is_ the noise source. We don't suppress thermal fluctuations—we _harvest_ them.

## Core Thesis

> **Interacting particle systems with pairwise repulsion are a universal computational primitive. Temperature selects what computation you're doing:**
>
> - **T → 0**: Optimization (gradient descent to minima)
> - **T ~ 0.1**: Sampling (Bayesian posterior inference)
> - **T → ∞**: Entropy generation (random number production)

These are not three different algorithms—they are **one algorithm** with a temperature parameter that controls the exploration/exploitation tradeoff.

## The Unified Update Equation

All three modes share the same Langevin dynamics update:

```
dx = -γ∇E(x)·dt + repulsion·dt + √(2γT·dt)·dW
```

Where:

- `∇E(x)` = gradient of energy/loss function (attraction to minima)
- `repulsion` = pairwise kernel gradient (keeps particles diverse)
- `T` = temperature (controls noise magnitude)
- `dW` = Wiener process (Brownian motion)

### How Temperature Controls Behavior

| Temperature | Dominant Term | Behavior                          | Output               |
| ----------- | ------------- | --------------------------------- | -------------------- |
| T → 0       | Gradient      | Particles flow downhill to minima | Optimized parameters |
| T ~ 0.1     | Balance       | Particles sample from exp(-E/T)   | Posterior samples    |
| T >> 1      | Noise         | Particles explore chaotically     | Random bits          |

## What Thermodynamic Sampling Can Do (That Gradient Descent Can't)

### 1. Multi-Modal Mode Discovery

Standard gradient descent converges to ONE local minimum. Thermodynamic sampling with SVGD repulsion finds ALL modes simultaneously.

**Demo**: `cargo run --release --features gpu --bin mode-discovery`

```
Energy landscape: E(x,y) = (x² - 4)² + (y² - 4)²
Known minima at: (-2,-2), (-2,+2), (+2,-2), (+2,+2)

Method 1: Gradient Descent (T → 0, no repulsion)
  Particles converged to 1 mode(s)

Method 2: Thermodynamic Sampling (T = 0.5, SVGD repulsion)
  Particles spread across 4 mode(s):
    (+2.00, +2.00): 25 particles
    (-2.00, +2.00): 24 particles
    (+2.00, -2.00): 26 particles
    (-2.00, -2.00): 25 particles

4D Hypercube Test: Discovered 16/16 modes of {-1,+1}^4
```

The SVGD repulsion term prevents mode collapse—particles push each other apart to cover the full landscape.

### 2. Bayesian Uncertainty Quantification

Instead of finding ONE weight vector, sample from the posterior p(weights|data) to get calibrated uncertainty estimates.

**Demo**: `cargo run --release --features gpu --bin bayesian-uncertainty`

```
Testing predictions with uncertainty from posterior samples:

    x1     x2   y_true     y_mean      y_std     region
--------------------------------------------------------------
  0.00   0.00      0.0     0.0603     0.0516   training  ← confident
  0.50   0.50      0.5     0.3996     0.4492     interp  ← uncertain
  2.00   0.00        ?     0.7449     0.3395     extrap  ← very uncertain
```

Key insight: uncertainty tells you WHERE the model is confident. Training points show low std (~0.05), extrapolation shows high std (~0.35).

### 3. Parallel Tempering (Replica Exchange)

Run multiple replicas at different temperatures. Hot replicas explore globally; cold replicas exploit locally. Periodic swaps transfer good solutions down the temperature ladder.

**Demo**: `cargo run --release --features gpu --bin parallel-tempering`

```
Parallel Tempering vs Standard Annealing on Rastrigin:

Standard Annealing:  Best = 9.18 (stuck in local minimum)
Parallel Tempering:  Best = 3.15 (found better basin)

Winner: Parallel Tempering by 6.03
```

### 4. Adaptive Annealing with Reheat

The `AdaptiveScheduler` automatically adjusts cooling rate based on optimization progress:

- **Convergence detection**: Cools faster when near optimum
- **Stall detection**: Slows cooling when stuck
- **Reheating**: Escapes local minima on deceptive landscapes

**Demo**: `cargo run --release --features gpu --bin adaptive-annealing`

On Schwefel (global minimum at 420.97, far from origin):

```
Fixed Schedule:    118.68 (stuck at local min)
Adaptive Schedule: 0.00   (found global min after 16 reheats)
```

## Implementation

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Compute Shader                        │
├─────────────────────────────────────────────────────────────┤
│  Pass 1: Compute pairwise repulsion (O(nK) subsampling)     │
│  Pass 2: Update particles (gradient + repulsion + noise)    │
│  Pass 3: Extract entropy (high-T mode only)                 │
└─────────────────────────────────────────────────────────────┘
```

### Performance Optimizations

**O(nK) Subsampling**: Reduces O(n²) repulsion to O(nK) with configurable K:

```
REPULSION SAMPLING COMPARISON (1000 particles, dim=4)
     Samples    Steps/sec      µs/step      Speedup
        skip      10827.2         92.4         21.1x
          64       3228.4        309.8          6.3x
        1000        482.2       2074.0          1.0x
```

### Technology Stack

- **Language**: Rust
- **GPU**: wgpu with WGSL compute shaders
- **Visualization**: iced GUI framework (optional `viz` feature)
- **Parallelism**: Up to 16,000 particles at 72+ steps/sec

## Expression DSL

Define custom loss functions using a composable DSL that compiles to GPU-accelerated WGSL:

```rust
use temper::expr::*;

// Griewank function: 1 + sum(x²/4000) - prod(cos(x/√(i+1)))
let griewank = const_(1.0)
    + sum_dims(|x, _| x.powi(2) / 4000.0)
    - prod_dims(|x, i| cos(x / sqrt(i + 1.0)));

// Create system with custom expression
let mut system = ThermodynamicSystem::with_expr(500, 4, 1.0, griewank);
```

**Demo**: `cargo run --release --features gpu --bin custom-expr-demo`

Available primitives:

- **Variables**: `var()`, `dim_index()`, `dim_count()`, `pi()`
- **Math**: `sin()`, `cos()`, `exp()`, `ln()`, `sqrt()`, `abs()`, `tanh()`
- **Operators**: `+`, `-`, `*`, `/`, `.powi()`, `.powf()`
- **Reductions**: `sum_dims(|x, i| ...)`, `prod_dims(|x, i| ...)`, `sum_pairs(|x, y| ...)`

Pre-built expressions: `sphere()`, `rastrigin()`, `rosenbrock()`, `ackley()`, `griewank()`, `levy()`, `michalewicz()`, `styblinski_tang()`, `dixon_price()`, `zakharov()`, and more.

## Empirical Validation

### Neural Network Training

| Test                  | Parameters   | Result                 |
| --------------------- | ------------ | ---------------------- |
| XOR MLP               | 9 (2→2→1)    | 100% accuracy          |
| Deep MLP (circles)    | 37 (2→4→4→1) | 100% accuracy          |
| Spiral classification | 9            | Successfully separates |

**Demo**: `cargo run --release --features gpu --bin deep-nn-benchmark`

### Optimization Benchmarks

| Function   | Dimension | Result                                 |
| ---------- | --------- | -------------------------------------- |
| Sphere     | N         | Converges to origin                    |
| Rosenbrock | N         | Finds (1,1,...,1)                      |
| Rastrigin  | 8D        | Adaptive beats fixed by 4.96           |
| Schwefel   | 2D        | Adaptive finds global min (16 reheats) |

### Entropy Quality (T >> 1)

**dieharder Statistical Tests** (all PASSED):

| Test              | p-value   | Assessment |
| ----------------- | --------- | ---------- |
| diehard_birthdays | 0.958     | PASSED     |
| diehard_rank_6x8  | 0.328     | PASSED     |
| sts_serial (1-16) | 0.11-0.99 | ALL PASSED |
| rgb_kstest_test   | 0.864     | PASSED     |

**`ent` Analysis** (1MB sample):

| Metric             | Value                  | Ideal  | Quality      |
| ------------------ | ---------------------- | ------ | ------------ |
| Entropy            | **7.999807** bits/byte | 8.0    | Near-perfect |
| Chi-square         | 28.13%                 | 10-90% | Pass         |
| Serial correlation | 0.0016                 | 0.0    | Near-zero    |

**Demo**: `cargo run --release --features gpu --bin thermodynamic-stream | dieharder -a -g 200`

## Hardware Implications

The noise paradox suggests a fundamentally different approach to AI hardware:

### Current Approach (Wasteful)

```
Physical Noise → Suppress → Deterministic Logic → PRNG → Synthetic Noise → AI
```

### Thermodynamic Approach (Efficient)

```
Physical Noise → Harvest → Thermodynamic Compute → AI
```

Potential implementations:

1. **Analog Oscillators**: Coupled LC circuits with thermal noise as the Langevin term
2. **Stochastic Digital ASIC**: True random number generators feeding particle updates
3. **Thermal Memory**: Exploit rather than correct memory bit flips at elevated temperature

See [HARDWARE.md](HARDWARE.md) for detailed hardware design concepts.

## Available Demos

| Binary                 | Description                      |
| ---------------------- | -------------------------------- |
| `benchmark`            | GPU performance scaling tests    |
| `deep-nn-benchmark`    | 37-param MLP on circles          |
| `optimizer-comparison` | Thermodynamic vs SGD vs Adam     |
| `parallel-tempering`   | Replica exchange demo            |
| `bayesian-sampling`    | Posterior sampling visualization |
| `bayesian-uncertainty` | Uncertainty quantification       |
| `mode-discovery`       | Multi-modal mode finding         |
| `adaptive-annealing`   | Fixed vs adaptive comparison     |
| `high-dim-benchmark`   | Dimension scaling tests          |
| `custom-expr-demo`     | Expression DSL showcase          |
| `thermodynamic-viz`    | Interactive visualization        |
| `rastrigin-annealing`  | Rastrigin visualization          |
| `schwefel-annealing`   | Schwefel visualization           |
| `thermodynamic-stream` | Entropy streaming to stdout      |
| `rng-demo`             | ThermodynamicRng usage           |
| `hyperparam-tuning`    | Hyperparameter optimization      |

## Public API

```rust
use temper::{
    ThermodynamicSystem,    // GPU-accelerated particle system
    ThermodynamicRng,       // RNG implementing rand_core::RngCore
    AdaptiveScheduler,      // Dimension-aware temperature scheduling
    LossFunction,           // Built-in loss functions
    ThermodynamicParticle,  // Particle state (pos, vel, energy)
    ThermodynamicMode,      // Operating mode enum
    ThermodynamicStats,     // System statistics
    RngCore,                // Re-exported from rand_core
};

use temper::expr::*;        // Expression DSL
```

## Connection to Existing Work

### Known Foundations

- **Langevin Dynamics** (1908): The update equation is standard statistical mechanics
- **Simulated Annealing** (1983): Temperature-controlled exploration for optimization
- **SVGD** (Liu & Wang, 2016): Particle-based variational inference with kernel repulsion
- **Parallel Tempering** (1986): Replica exchange for enhanced sampling

### Novel Contributions

1. **Unified framework**: Same codebase performs optimization, sampling, AND entropy generation
2. **Temperature as computation selector**: Smooth transitions between modes
3. **Noise paradox insight**: Harvesting rather than fighting physical noise
4. **Expression DSL**: Composable loss functions compiling to GPU shaders
5. **Adaptive scheduling**: Dimension-aware annealing with automatic reheat

## Future Directions

1. **Diffusion models**: The denoising process is Langevin dynamics—direct connection
2. **Larger neural networks**: Scale beyond 37 parameters to practical networks
3. **Python bindings**: PyO3 wrapper for ML ecosystem integration
4. **WebGPU demo**: Browser-based interactive visualization
5. **Hardware prototypes**: Analog or stochastic digital implementations

## Conclusion

We have demonstrated that a single interacting particle system, controlled solely by temperature, can correctly perform optimization, Bayesian sampling, and entropy generation. The SVGD repulsion term is crucial—it prevents mode collapse and enables capabilities impossible with gradient descent alone:

- **Find ALL modes** of multimodal distributions
- **Quantify uncertainty** through posterior sampling
- **Escape local minima** via parallel tempering and adaptive reheat
- **Generate entropy** from chaotic high-temperature dynamics

The noise paradox reveals the inefficiency of current AI hardware: we suppress physical noise, then simulate it back. Thermodynamic computing offers a path to hardware that _embraces_ noise as a computational resource.

Temperature isn't just a hyperparameter—it's a **computation selector** that smoothly interpolates between deterministic optimization and stochastic exploration. This is one algorithm, not three.

---

_Temper: GPU-accelerated unified thermodynamic particle systems._
_Run `cargo run --release --features gpu --bin benchmark` to see it in action._
