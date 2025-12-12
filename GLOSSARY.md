# Glossary of Thermodynamic Computing

A comprehensive reference for all concepts, algorithms, and terminology used in this codebase.

## Core Physics & Mathematics

### Langevin Dynamics
A stochastic differential equation describing how a particle moves through a potential energy landscape while being buffeted by thermal fluctuations. Named after Paul Langevin (1908).

**Overdamped form** (used in this codebase):
```
dx = -γ∇E(x)·dt + √(2γT)·dW
```

Where:
- `γ` = friction coefficient (controls step size)
- `∇E(x)` = gradient of energy/loss function
- `T` = temperature
- `dW` = Wiener process increment (Brownian noise)

In the overdamped limit, inertia is negligible compared to friction, simplifying the second-order equation to first-order. This is also called **Brownian dynamics**.

**Reference**: [Langevin equation - Wikipedia](https://en.wikipedia.org/wiki/Langevin_equation)

### Boltzmann Distribution
The probability distribution for a system in thermal equilibrium at temperature T:

```
p(x) ∝ exp(-E(x) / kT)
```

Where k is Boltzmann's constant (often set to 1 in computational contexts). At thermal equilibrium, Langevin dynamics samples from this distribution.

**Key insight**: Lower energy states are exponentially more probable. Temperature controls the "sharpness" of this preference—low T concentrates probability on minima, high T spreads it broadly.

**Reference**: [Boltzmann distribution - Wikipedia](https://en.wikipedia.org/wiki/Boltzmann_distribution)

### Wiener Process (Brownian Motion)
A continuous-time stochastic process W(t) with:
- W(0) = 0
- Independent increments
- W(t+h) - W(t) ~ Normal(0, h)

The `dW` in stochastic differential equations represents an infinitesimal increment of this process. Sample paths are continuous but nowhere differentiable.

**Reference**: [Wiener process - Wikipedia](https://en.wikipedia.org/wiki/Wiener_process)

### Score Function
The gradient of the log probability density:
```
s(x) = ∇log p(x)
```

For a Boltzmann distribution p(x) ∝ exp(-E(x)/T):
```
s(x) = -∇E(x) / T
```

This connects energy-based models to score-based generative models.

**Reference**: [Yang Song's blog on score-based models](https://yang-song.net/blog/2021/score/)

---

## Optimization Algorithms

### Simulated Annealing
Optimization by analogy with metallurgical annealing. Start at high temperature (broad exploration), gradually cool to low temperature (exploitation of minima).

**Key idea**: High temperature allows escaping local minima; slow cooling lets the system settle into global minima.

**Origin**: Kirkpatrick, Gelatt, & Vecchi (1983). "Optimization by Simulated Annealing." *Science* 220(4598), 671-680.

**Reference**: [Science paper](https://www.science.org/doi/10.1126/science.220.4598.671)

### Parallel Tempering (Replica Exchange MCMC)
Run multiple copies ("replicas") of the system at different temperatures simultaneously. Periodically attempt to swap configurations between adjacent temperature levels.

**Swap acceptance probability** (Metropolis criterion):
```
P_accept = min(1, exp((E_i - E_j)(1/T_i - 1/T_j)))
```

Hot replicas explore globally; cold replicas exploit locally. Swaps allow good solutions to propagate from hot to cold.

**Origin**: Swendsen & Wang (1986). "Replica Monte Carlo simulation of spin-glasses." *Physical Review Letters* 57(21), 2607.

**Reference**: [PRL paper](https://link.aps.org/doi/10.1103/PhysRevLett.57.2607)

### Metropolis-Hastings Algorithm
MCMC method for sampling from a probability distribution. Given current state x and proposed state x':

**Acceptance probability**:
```
α = min(1, [π(x') q(x|x')] / [π(x) q(x'|x)])
```

Where π is the target distribution and q is the proposal distribution.

For symmetric proposals (q(x'|x) = q(x|x')), this simplifies to:
```
α = min(1, π(x')/π(x))
```

**Reference**: [Metropolis-Hastings - Wikipedia](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm)

### Gradient Descent
Iterative optimization: move in the direction opposite to the gradient.
```
x_{t+1} = x_t - η∇E(x_t)
```

The T→0 limit of Langevin dynamics (noise term vanishes).

### Adaptive Annealing
Dynamic adjustment of temperature schedule based on optimization progress:
- **Convergence detection**: Cool faster when approaching optimum
- **Stall detection**: Slow down or reheat when stuck
- **Reheating**: Escape local minima by temporarily increasing temperature

---

## Particle Methods

### SVGD (Stein Variational Gradient Descent)
Particle-based variational inference that combines gradient descent with inter-particle repulsion:

```
x_i ← x_i + ε · (1/n) Σ_j [k(x_j, x_i)∇E(x_j) + ∇_{x_j}k(x_j, x_i)]
```

The first term follows gradients; the second term (kernel gradient) pushes particles apart.

**Key property**: Prevents mode collapse—particles spread to cover all modes of the target distribution.

**Kernel**: Typically RBF (Gaussian):
```
k(x, y) = exp(-||x-y||² / 2h²)
```

**Origin**: Liu & Wang (2016). "Stein variational gradient descent: A general purpose Bayesian inference algorithm." *NeurIPS*.

**Reference**: [NeurIPS paper](https://proceedings.neurips.cc/paper/2016/hash/b3ba8f1bee1238a2f37603d90b58898d-Abstract.html)

### RBF Kernel (Radial Basis Function)
```
k(x, y) = exp(-||x-y||² / 2h²)
```

Properties:
- Value decreases with distance (similarity measure)
- Ranges from 0 (infinite distance) to 1 (x = y)
- Bandwidth h controls the "reach" of the kernel

Also known as Gaussian kernel or squared exponential kernel.

**Reference**: [RBF kernel - Wikipedia](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)

---

## Benchmark Functions

### Sphere Function
```
f(x) = Σ x_i²
```
- **Minimum**: f(0,...,0) = 0
- **Character**: Simple convex function, easy baseline
- **Use**: Sanity check for optimization algorithms

### Rosenbrock Function (Banana Valley)
```
f(x) = Σ [100(x_{i+1} - x_i²)² + (1 - x_i)²]
```
- **Minimum**: f(1,...,1) = 0
- **Character**: Long, narrow, curved valley
- **Challenge**: Finding the valley is easy; converging along it is hard

Named for Howard Rosenbrock (1960). Tests ability to navigate narrow curved regions.

**Reference**: [Rosenbrock function - Wikipedia](https://en.wikipedia.org/wiki/Rosenbrock_function)

### Rastrigin Function
```
f(x) = 10n + Σ [x_i² - 10cos(2πx_i)]
```
- **Minimum**: f(0,...,0) = 0
- **Character**: Highly multimodal (~10^n local minima in a grid pattern)
- **Challenge**: Many local minima trap gradient-based methods

Named for L.A. Rastrigin (1974). Classic test for global optimization.

**Reference**: [Rastrigin function - Wikipedia](https://en.wikipedia.org/wiki/Rastrigin_function)

### Ackley Function
```
f(x) = -20exp(-0.2√(Σx_i²/n)) - exp(Σcos(2πx_i)/n) + 20 + e
```
- **Minimum**: f(0,...,0) = 0
- **Character**: Nearly flat outer region with sharp central hole
- **Challenge**: "Needle in a haystack"—requires broad exploration then precise exploitation

Named for David Ackley (1987).

**Reference**: [Ackley function - Wikipedia](https://en.wikipedia.org/wiki/Ackley_function)

### Schwefel Function
```
f(x) = 418.9829n - Σ x_i·sin(√|x_i|)
```
- **Minimum**: f(420.9687,...,420.9687) = 0
- **Character**: Global minimum far from origin; local minima point wrong direction
- **Challenge**: Highly deceptive—gradients mislead toward wrong regions

**Reference**: [Schwefel function](https://www.sfu.ca/~ssurjano/schwef.html)

---

## Neural Network Concepts

### MLP (Multi-Layer Perceptron)
Feedforward neural network with:
- Input layer
- One or more hidden layers with nonlinear activations
- Output layer

**XOR Network** (this codebase): 2→2→1 with 9 parameters
**Deep Network** (this codebase): 2→4→4→1 with 37 parameters

### Activation Functions
- **tanh**: Output in (-1, 1), smooth gradient
- **sigmoid**: Output in (0, 1), for probabilities: σ(x) = 1/(1+e^(-x))
- **ReLU**: max(0, x), sparse activations

### Binary Cross-Entropy Loss
```
L = -[y·log(p) + (1-y)·log(1-p)]
```
For binary classification. Measures divergence between predicted probability p and true label y.

### Softmax
Converts raw scores to probabilities:
```
softmax(z)_i = exp(z_i) / Σ exp(z_j)
```

---

## Connection to Diffusion Models

### Score-Based Generative Models
Learn the score function s(x) = ∇log p(x) and use Langevin dynamics to sample.

**Key equivalence**:
- Diffusion forward: Add noise progressively
- Diffusion reverse: Langevin dynamics with decreasing noise scale
- Temperature annealing = reverse diffusion with analytical score

**Reference**: [Score-based generative modeling](https://yang-song.net/blog/2021/score/)

---

## GPU & Implementation

### WGPU
Cross-platform GPU abstraction layer for Rust. Implements the WebGPU standard, supporting Metal (macOS), Vulkan (Linux/Windows), DX12 (Windows).

### WGSL (WebGPU Shading Language)
GPU shader language for WebGPU. Supports:
- Compute shaders for parallel computation
- `f16` (half-precision) for memory bandwidth optimization
- Storage buffers for GPU-CPU data transfer

### Compute Shader
GPU program that runs parallel computation. In this codebase:
1. `compute_repulsion`: Calculate SVGD repulsion forces
2. `update_particles`: Apply Langevin dynamics update

### f16 (Half-Precision Float)
16-bit floating point. Trades precision for:
- 50% memory bandwidth reduction
- Faster computation on GPUs with native f16 support

### Workgroup
Group of GPU threads that execute together. Typical size: 64 threads.

---

## Statistical Concepts

### Effective Sample Size (ESS)
Accounts for correlation between samples:
```
ESS = (Σw_i)² / Σw_i²
```
Where w_i = exp(-(E_i - E_min)/T). Measures quality of Monte Carlo samples.

### Mode Collapse
When an algorithm converges to a single mode of a multimodal distribution, ignoring other modes. SVGD repulsion prevents this.

### Ergodicity
Property that time averages equal ensemble averages. At high temperature, Langevin dynamics is ergodic—particles eventually visit all regions of state space.

---

## Diversity Metrics (implemented in codebase)

- **Mean Pairwise Distance**: Average Euclidean distance between particle pairs
- **Distance Std Dev**: Spread of pairwise distances
- **Energy Variance**: Variance of loss values across particles
- **Estimated Mode Count**: Number of distinct low-energy clusters
- **Coverage**: Fraction of search space bounding box occupied

---

## Key Parameters

| Parameter | Symbol | Role |
|-----------|--------|------|
| Temperature | T | Controls exploration vs exploitation |
| Friction | γ | Step size / learning rate factor |
| Time step | dt | Integration step size |
| Kernel bandwidth | h | SVGD repulsion range |
| Repulsion strength | - | SVGD force magnitude |
| Repulsion samples | K | Number of particles for O(nK) SVGD |

---

## References

1. Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671-680.

2. Liu, Q., & Wang, D. (2016). Stein variational gradient descent: A general purpose Bayesian inference algorithm. *NeurIPS*.

3. Swendsen, R. H., & Wang, J. S. (1986). Replica Monte Carlo simulation of spin-glasses. *Physical Review Letters*, 57(21), 2607.

4. Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *NeurIPS*.

5. Rosenbrock, H. H. (1960). An automatic method for finding the greatest or least value of a function. *The Computer Journal*, 3(3), 175-184.

6. Rastrigin, L. A. (1974). Systems of extremal control. *Nauka*.
