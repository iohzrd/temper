# Thermodynamic Computing Hardware

How custom silicon could efficiently implement the Temper algorithm natively.

## Core Insight

The unified update equation:

```
dx = -γ∇E(x)·dt + repulsion·dt + √(2γT·dt)·dW
```

This is **Langevin dynamics** - the same physics that governs particles in thermal equilibrium. We're simulating on GPUs what physical systems do naturally.

**The opportunity**: Build hardware where the computation _is_ the physics, not a simulation of it.

---

## Approach 1: Analog Thermodynamic Computing

### The Insight

We simulate `dx = -∇E·dt + √(2T)·dW` digitally. But real physical systems _already obey this equation_ - thermal noise is free, gradient descent is just relaxation to equilibrium.

### Design

| Component            | Physical Implementation                                           |
| -------------------- | ----------------------------------------------------------------- |
| **Particles**        | Coupled oscillators (LC circuits, MEMS resonators, optical modes) |
| **Energy landscape** | Programmable coupling strengths between oscillators               |
| **Temperature**      | Actual temperature OR adjustable noise injection                  |
| **Gradient descent** | Natural relaxation to lowest-energy state                         |
| **Readout**          | Measure oscillator phases/amplitudes                              |

### Advantages

- **Free noise generation** - Johnson-Nyquist thermal noise
- **Continuous-time dynamics** - No dt discretization errors
- **Massive parallelism** - Millions of coupled oscillators possible
- **Energy efficient** - Computation = relaxation to equilibrium
- **Mode switching** - Temperature dial transitions optimize↔sample↔entropy

### Existing Work

| System                      | Organization | Scale       | Speed          |
| --------------------------- | ------------ | ----------- | -------------- |
| Coherent Ising Machines     | Stanford/NTT | 100k+ spins | µs solve times |
| Probabilistic bits (p-bits) | Purdue       | 8+ bits     | ns flips       |
| Oscillator computing        | Georgia Tech | 1000s       | MHz operation  |

---

## Approach 2: Stochastic Digital ASIC

If staying digital, optimize for our specific computation pattern.

### Per-Particle Processing Unit

```
┌─────────────────────────────────────────────────────┐
│  Position registers [64 dimensions, 16-bit float]   │
│  Velocity registers [64 dimensions, 16-bit float]   │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │  Gradient   │  │    Noise    │  │  Repulsion │  │
│  │    Unit     │  │  Generator  │  │   Accumul  │  │
│  │  (LUT/FMA)  │  │   (LFSR)    │  │            │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬─────┘  │
│         └────────────┬───┴─────────────────┘       │
│                ┌─────▼─────┐                       │
│                │  Langevin │                       │
│                │   Update  │  (FMA units)          │
│                │  x += ... │                       │
│                └───────────┘                       │
└─────────────────────────────────────────────────────┘
                    ↕ Repulsion interconnect (systolic)
```

### Key Optimizations

| Optimization            | Rationale                                                  |
| ----------------------- | ---------------------------------------------------------- |
| **Reduced precision**   | 16-bit or 8-bit floats sufficient (noise dominates)        |
| **Hardwired gradients** | Common functions (Rastrigin, etc.) as lookup tables        |
| **Streaming repulsion** | Systolic array for O(nK) pairwise computation              |
| **On-chip TRNG**        | Ring oscillator jitter for true randomness                 |
| **Fused Langevin op**   | Single instruction: `x += γ*(grad + repulsion)*dt + noise` |

### Estimated Performance

| Metric    | GPU (current) | Custom ASIC (projected) |
| --------- | ------------- | ----------------------- |
| Particles | 16k           | 1M+                     |
| Steps/sec | 72            | 10k+                    |
| Power     | 300W          | 10W                     |
| Latency   | ms            | µs                      |

---

## Approach 3: Thermal Memory (Radical)

### Concept

**What if the "particles" were actual electrons in a potential well?**

- Quantum dots or single-electron transistors as particles
- Gate voltages define energy landscape
- Real thermal fluctuations provide noise
- Cooling = annealing, heating = exploration
- Read electron positions capacitively

### Implementation

```
        Gate Array (defines E(x))
        ↓   ↓   ↓   ↓   ↓   ↓
    ┌───┴───┴───┴───┴───┴───┴───┐
    │  ·   ·   ·   ·   ·   ·   · │  ← Electrons in 2DEG
    │    ·   ·   ·   ·   ·   ·   │
    │  ·   ·   ·   ·   ·   ·   · │
    └───────────────────────────┘
        ↑
    Capacitive readout array
```

### Temperature as Control Knob

| Physical Temperature | Computational Mode               |
| -------------------- | -------------------------------- |
| 4K (cryogenic)       | Optimize - minimal thermal noise |
| 77K (liquid N₂)      | Sample - moderate exploration    |
| 300K (room temp)     | Entropy - maximum randomness     |

This is essentially **physical simulated annealing** without simulation.

---

## The Killer Feature: Unified Mode Switching

All three approaches share this property:

```
                    TEMPERATURE
         Cold ←─────────────────────→ Hot
           │                           │
           ▼                           ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ OPTIMIZE │   │  SAMPLE  │   │ ENTROPY  │
    │ T → 0    │   │ T ~ 0.1  │   │ T >> 1   │
    │          │   │          │   │          │
    │ Find     │   │ Bayesian │   │ Random   │
    │ minimum  │   │ posterior│   │ bits     │
    └──────────┘   └──────────┘   └──────────┘
```

**Same circuit. Same particles. Different temperature.**

No reconfiguration, no mode switching logic - just turn the temperature dial.

---

## Comparison: Digital vs Analog vs Thermal

| Aspect                  | GPU (Digital) | Analog Oscillators  | Thermal Memory |
| ----------------------- | ------------- | ------------------- | -------------- |
| **Noise source**        | PRNG          | Injected/thermal    | True thermal   |
| **Precision**           | 32-bit float  | ~8-10 bit effective | Continuous     |
| **Parallelism**         | 10⁴ threads   | 10⁵-10⁶ oscillators | 10⁸+ electrons |
| **Energy/op**           | ~pJ           | ~fJ                 | ~kT (~26 meV)  |
| **Programmability**     | Full          | Coupling matrix     | Gate voltages  |
| **Temperature control** | Parameter     | Noise amplitude     | Literal        |
| **Maturity**            | Production    | Research            | Speculative    |

---

## What Would Make This Practical?

### Required Capabilities

1. **Programmable coupling matrix** - Define arbitrary energy landscapes
2. **Fast temperature control** - For annealing schedules (µs timescale)
3. **Efficient parallel readout** - Sample all particle states simultaneously
4. **Hybrid interface** - Digital control plane, analog/thermal compute core

### Target Applications

| Application                | Why Thermodynamic Hardware?              |
| -------------------------- | ---------------------------------------- |
| Combinatorial optimization | Natural fit for Ising/QUBO               |
| Bayesian neural networks   | Sample posteriors, not point estimates   |
| Drug discovery             | Sample molecular conformations           |
| Cryptographic RNG          | True thermal randomness                  |
| Reinforcement learning     | Exploration-exploitation via temperature |

---

## Related Research

### Academic

- **Coherent Ising Machines** - Yamamoto group (Stanford), Inagaki et al. (NTT)
- **Probabilistic Computing** - Camsari et al. (Purdue)
- **Thermodynamic Computing** - Boyd et al., Conte et al.
- **Oscillator-Based Computing** - Roychowdhury group (Georgia Tech)

### Industry

- **Normal Computing** - Thermodynamic AI chips
- **Lightmatter** - Optical computing (related architecture)
- **D-Wave** - Quantum annealing (different physics, similar concept)
- **Cerebras** - Wafer-scale integration (digital, but relevant for parallelism)

---

## Conclusion

The Temper algorithm demonstrates that optimization, sampling, and entropy generation are unified under Langevin dynamics. This isn't just a software insight - it's a **hardware architecture principle**.

Custom silicon could implement this natively by:

1. Using **real thermal noise** instead of PRNGs
2. Encoding **energy landscapes** in physical coupling
3. Controlling **temperature** as the primary compute parameter
4. Achieving **massive parallelism** through physics

The result: orders of magnitude improvement in energy efficiency and speed for thermodynamic computation, with the elegant property that a single device smoothly transitions between optimizer, sampler, and entropy source.

---

_This document accompanies the [Temper](README.md) software implementation._
