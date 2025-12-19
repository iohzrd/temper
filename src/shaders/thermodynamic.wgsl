// Unified Thermodynamic Particle System - HMC Implementation
//
// Uses Hamiltonian Monte Carlo (HMC) with leapfrog integration.
// Temperature controls behavior across three modes:
//
// T >> 1   : Entropy mode    - Large momenta, chaotic exploration for randomness
// T ~ 0.1  : Sampling mode   - Balanced exploration, efficient posterior sampling
// T → 0    : Optimize mode   - Small momenta, momentum-based gradient descent
//
// The key insight: These are NOT three different algorithms, but ONE algorithm
// with a temperature parameter that controls the exploration/exploitation tradeoff.
//
// MEMORY LAYOUT: Structure of Arrays (SoA) with dimension-major ordering
// ========================================================================
// Instead of Array of Structures (AoS): [particle0: all fields][particle1: all fields]...
// We use Structure of Arrays (SoA):     [all positions][all velocities][scalars]...
//
// Within positions/velocities, we use DIMENSION-MAJOR layout:
//   positions[d * particle_count + idx] = position of particle idx in dimension d
//
// This gives PERFECT memory coalescing when all threads in a warp access the
// same dimension d simultaneously (which happens in the leapfrog inner loop).
// Warp threads access consecutive memory: pos[d*N+0], pos[d*N+1], ..., pos[d*N+31]

// Enable f16 support for reduced memory bandwidth (~50% memory reduction)
enable f16;

// Per-particle scalar data (accessed once per particle per step)
// Kept as struct since these aren't accessed in the dimension loop
struct ParticleScalars {
    energy: f32,              // Potential energy U(q) = loss value
    kinetic_energy: f32,      // Kinetic energy K(p) = |p|²/2m
    mass: f32,                // Particle mass m
    entropy_bits: u32,        // Accumulated entropy bits (for extraction)
}

// Legacy Particle struct for compatibility with proposal buffer
// TODO: Migrate proposal buffer to SoA in future optimization
struct Particle {
    pos: array<f16, 256>,
    vel: array<f16, 256>,
    energy: f32,
    kinetic_energy: f32,
    mass: f32,
    entropy_bits: u32,
}

// Helper to read position from SoA layout as f32
fn get_pos_soa(idx: u32, d: u32, pc: u32) -> f32 {
    return f32(positions[d * pc + idx]);
}

// Helper to read velocity from SoA layout as f32
fn get_vel_soa(idx: u32, d: u32, pc: u32) -> f32 {
    return f32(velocities[d * pc + idx]);
}

// Helper to write position to SoA layout
fn set_pos_soa(idx: u32, d: u32, pc: u32, val: f32) {
    positions[d * pc + idx] = f16(val);
}

// Helper to write velocity to SoA layout
fn set_vel_soa(idx: u32, d: u32, pc: u32, val: f32) {
    velocities[d * pc + idx] = f16(val);
}

// Helper to get position array as f32 for loss functions (from SoA)
fn get_pos_array_soa(idx: u32, dim: u32, pc: u32) -> array<f32, 256> {
    var result: array<f32, 256>;
    for (var i = 0u; i < dim; i = i + 1u) {
        result[i] = f32(positions[i * pc + idx]);
    }
    return result;
}

// Legacy helpers for proposal buffer compatibility
fn get_pos(p: Particle, d: u32) -> f32 {
    return f32(p.pos[d]);
}

fn get_pos_array(p: Particle, dim: u32) -> array<f32, 256> {
    var result: array<f32, 256>;
    for (var i = 0u; i < dim; i = i + 1u) {
        result[i] = f32(p.pos[i]);
    }
    return result;
}

struct Uniforms {
    particle_count: u32,
    dim: u32,
    temperature: f32,        // THE KEY PARAMETER: controls mode
    seed: u32,
    mode: u32,               // 0=optimize, 1=sample, 2=entropy
    loss_fn: u32,            // Loss function selector
    leapfrog_steps: u32,     // L - number of leapfrog steps per HMC iteration
    step_size: f32,          // ε - leapfrog integration step size
    mass: f32,               // m - particle mass for kinetic energy
    _padding: u32,           // Padding for alignment
    pos_min: f32,            // Position domain minimum (e.g., 0.0 for images)
    pos_max: f32,            // Position domain maximum (e.g., 1.0 for images)
    _padding2: vec2<u32>,    // Padding for 16-byte alignment
}

// Training data (same neural net task)
const TRAIN_X: array<f32, 10> = array<f32, 10>(
    -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5
);
const TRAIN_Y: array<f32, 10> = array<f32, 10>(
    -1.93, -1.79, -1.52, -0.93, 0.0, 0.93, 1.52, 1.79, 1.93, 1.97
);
const TRAIN_SIZE: u32 = 10u;

// SoA buffers - dimension-major layout for optimal GPU coalescing
// Access pattern: buffer[dim * particle_count + particle_idx]
@group(0) @binding(0) var<storage, read_write> positions: array<f16>;          // [dim][particle]
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read_write> velocities: array<f16>;         // [dim][particle]
@group(0) @binding(3) var<storage, read_write> scalars: array<ParticleScalars>; // Per-particle scalars
@group(0) @binding(4) var<storage, read_write> entropy_output: array<u32>;
@group(0) @binding(5) var<storage, read_write> proposal_pos: array<f16>;       // Proposal positions [dim][particle]
@group(0) @binding(6) var<storage, read_write> proposal_vel: array<f16>;       // Proposal velocities [dim][particle]
@group(0) @binding(7) var<storage, read_write> proposal_scalars: array<ParticleScalars>; // Proposal scalars
@group(0) @binding(8) var<storage, read_write> accept_flags: array<u32>;       // 1 = accepted, 0 = rejected

// Hash function for pseudo-random numbers
fn hash(seed: u32) -> u32 {
    var x = seed;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn rand(seed: u32) -> f32 {
    return f32(hash(seed) & 0xFFFFFFu) / 16777216.0;
}

fn randn(seed1: u32, seed2: u32) -> f32 {
    let u1 = max(rand(seed1), 0.0001);
    let u2 = rand(seed2);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2);
}

// ============================================================================
// FAST MATH CONSTANTS AND APPROXIMATIONS
// These provide 1.5-3x speedup with acceptable accuracy for optimization
// ============================================================================
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const INV_TWO_PI: f32 = 0.15915494309;

// Fast sin approximation using Bhaskara I's formula (7th century!)
// Error < 1.8% over full period, much faster than built-in
fn fast_sin(x: f32) -> f32 {
    // Normalize to [0, 2π]
    var t = x * INV_TWO_PI;
    t = t - floor(t);  // Now in [0, 1]
    t = t * TWO_PI;    // Back to [0, 2π]

    // Shift to [-π, π] for better approximation
    if t > PI {
        t = t - TWO_PI;
    }

    // Bhaskara approximation: sin(x) ≈ 16x(π-x) / (5π² - 4x(π-x))
    let t2 = t * (PI - abs(t));
    return 4.0 * t2 / (5.0 * PI * PI - 4.0 * abs(t2)) * sign(t) * sign(PI - abs(t));
}

// Fast cos using sin identity
fn fast_cos(x: f32) -> f32 {
    return fast_sin(x + PI * 0.5);
}

// Fast exp approximation (clamped for stability)
fn fast_exp(x: f32) -> f32 {
    let x_clamped = clamp(x, -20.0, 20.0);
    return exp(x_clamped);
}

// Fast sqrt using hardware inverseSqrt
fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    return x * inverseSqrt(x);
}

// ============================================================================
// F16 COMPUTE HELPERS
// True f16 arithmetic for position updates - faster on GPUs with f16 acceleration
// ============================================================================

// Random number in f16
fn rand_f16(seed: u32) -> f16 {
    return f16(rand(seed));
}

// Box-Muller in f16 (less precise but faster)
fn randn_f16(seed1: u32, seed2: u32) -> f16 {
    let u1 = max(rand_f16(seed1), f16(0.001));
    let u2 = rand_f16(seed2);
    // Use f32 for transcendentals, convert result to f16
    return f16(sqrt(-2.0 * log(f32(u1))) * cos(6.283185307 * f32(u2)));
}

// Fast sin in f16 (Bhaskara approximation)
fn fast_sin_f16(x: f16) -> f16 {
    let pi = f16(3.14159);
    let two_pi = f16(6.28318);
    let inv_two_pi = f16(0.159155);

    var t = x * inv_two_pi;
    t = t - floor(t);
    t = t * two_pi;

    if t > pi {
        t = t - two_pi;
    }

    let t2 = t * (pi - abs(t));
    return f16(4.0) * t2 / (f16(5.0) * pi * pi - f16(4.0) * abs(t2)) * sign(t) * sign(pi - abs(t));
}

fn fast_cos_f16(x: f16) -> f16 {
    return fast_sin_f16(x + f16(1.5708));
}

// ============================================================================

// Stub functions for custom expressions - these are overridden when using with_expr()
// They must exist for the shader to compile even when LossFunction::Custom isn't used
fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    // Default: sphere function
    var sum = 0.0;
    for (var i = 0u; i < dim; i = i + 1u) {
        sum = sum + pos[i] * pos[i];
    }
    return sum;
}

fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    // Numerical gradient
    let eps = 0.001;
    var pos_plus = pos;
    var pos_minus = pos;
    pos_plus[d_idx] = pos[d_idx] + eps;
    pos_minus[d_idx] = pos[d_idx] - eps;
    return (custom_loss(pos_plus, dim) - custom_loss(pos_minus, dim)) / (2.0 * eps);
}

// 2D neural net: y = w2 * tanh(w1 * x)
fn nn_loss_2d(w1: f32, w2: f32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < TRAIN_SIZE; i = i + 1u) {
        let pred = w2 * tanh(w1 * TRAIN_X[i]);
        let err = pred - TRAIN_Y[i];
        sum = sum + err * err;
    }
    return sum / f32(TRAIN_SIZE);
}

fn nn_gradient_2d(w1: f32, w2: f32) -> vec2<f32> {
    var dw1 = 0.0;
    var dw2 = 0.0;
    for (var i = 0u; i < TRAIN_SIZE; i = i + 1u) {
        let x = TRAIN_X[i];
        let a = tanh(w1 * x);
        let pred = w2 * a;
        let err = pred - TRAIN_Y[i];
        let td = 1.0 - a * a;
        dw1 = dw1 + err * w2 * td * x;
        dw2 = dw2 + err * a;
    }
    let n = f32(TRAIN_SIZE);
    return vec2<f32>(2.0 * dw1 / n, 2.0 * dw2 / n);
}

// N-dimensional multimodal loss function
// Has 2^(dim/2) global minima at combinations of ±1.5 in each pair of dimensions
// This generalizes the 2D neural net's two minima to higher dimensions
fn multimodal_loss_nd(pos: array<f32, 256>, dim: u32) -> f32 {
    var loss = 0.0;
    // Sum over pairs of dimensions
    for (var d = 0u; d < dim; d = d + 2u) {
        let x = pos[d];
        let y = pos[min(d + 1u, dim - 1u)];
        // Each pair contributes a 2-minima landscape like the 2D neural net
        // Minima at approximately (±1.5, ±2.0) scaled by dimension
        let scale = 1.0 / (1.0 + f32(d) * 0.1);
        let target_x = 1.5 * scale;
        let target_y = 2.0 * scale;
        // Loss is minimum of distance to positive and negative optima
        let d_pos = (x - target_x) * (x - target_x) + (y - target_y) * (y - target_y);
        let d_neg = (x + target_x) * (x + target_x) + (y + target_y) * (y + target_y);
        loss = loss + min(d_pos, d_neg) * 0.5;
    }
    return loss;
}

// Gradient of N-dimensional multimodal loss
fn multimodal_gradient_nd(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    let pair_idx = d_idx / 2u;
    let in_pair = d_idx % 2u;
    let d_base = pair_idx * 2u;

    if d_base >= dim {
        return 0.0;
    }

    let x = pos[d_base];
    let y = pos[min(d_base + 1u, dim - 1u)];

    let scale = 1.0 / (1.0 + f32(d_base) * 0.1);
    let target_x = 1.5 * scale;
    let target_y = 2.0 * scale;

    let d_pos = (x - target_x) * (x - target_x) + (y - target_y) * (y - target_y);
    let d_neg = (x + target_x) * (x + target_x) + (y + target_y) * (y + target_y);

    // Gradient toward the closer optimum
    if d_pos < d_neg {
        if in_pair == 0u {
            return (x - target_x);
        } else {
            return (y - target_y);
        }
    } else {
        if in_pair == 0u {
            return (x + target_x);
        } else {
            return (y + target_y);
        }
    }
}

// ============================================================================
// REAL MLP NEURAL NETWORK - XOR CLASSIFICATION
// ============================================================================
// Network architecture: 2 inputs -> 2 hidden (tanh) -> 1 output (sigmoid)
// Parameters (9 total):
//   pos[0..3] = W1 (2x2 input->hidden weights)
//   pos[4..5] = b1 (2 hidden biases)
//   pos[6..7] = W2 (2x1 hidden->output weights)
//   pos[8]    = b2 (1 output bias)

// XOR training data
const XOR_X: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0)
);
const XOR_Y: array<f32, 4> = array<f32, 4>(0.0, 1.0, 1.0, 0.0);

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn mlp_xor_forward(pos: array<f32, 256>, input: vec2<f32>) -> f32 {
    // Layer 1: input (2) -> hidden (2) with tanh
    let h0 = tanh(pos[0] * input.x + pos[1] * input.y + pos[4]);
    let h1 = tanh(pos[2] * input.x + pos[3] * input.y + pos[5]);

    // Layer 2: hidden (2) -> output (1) with sigmoid
    let out = sigmoid(pos[6] * h0 + pos[7] * h1 + pos[8]);
    return out;
}

fn mlp_xor_loss(pos: array<f32, 256>) -> f32 {
    var total_loss = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let pred = mlp_xor_forward(pos, XOR_X[i]);
        let label = XOR_Y[i];
        // Binary cross-entropy loss
        let eps = 0.0001;
        let p = clamp(pred, eps, 1.0 - eps);
        total_loss = total_loss - (label * log(p) + (1.0 - label) * log(1.0 - p));
    }
    return total_loss / 4.0;
}

fn mlp_xor_gradient(pos: array<f32, 256>, d_idx: u32) -> f32 {
    // Numerical gradient (more stable for complex networks)
    let eps = 0.001;
    var pos_plus = pos;
    var pos_minus = pos;
    pos_plus[d_idx] = pos[d_idx] + eps;
    pos_minus[d_idx] = pos[d_idx] - eps;
    return (mlp_xor_loss(pos_plus) - mlp_xor_loss(pos_minus)) / (2.0 * eps);
}

// ============================================================================
// SPIRAL CLASSIFICATION (harder non-linear problem)
// ============================================================================
// Two interleaved spirals - classic ML benchmark
const SPIRAL_SIZE: u32 = 20u;

fn spiral_point(idx: u32, cls: u32) -> vec2<f32> {
    let r = f32(idx) / f32(SPIRAL_SIZE) * 3.0;
    let theta = f32(idx) / f32(SPIRAL_SIZE) * 4.0 * PI + f32(cls) * PI;
    return vec2<f32>(r * fast_cos(theta), r * fast_sin(theta));
}

fn mlp_spiral_loss(pos: array<f32, 256>) -> f32 {
    var total_loss = 0.0;

    // Sample points from both spirals
    for (var i = 0u; i < SPIRAL_SIZE; i = i + 1u) {
        // Class 0 spiral
        let p0 = spiral_point(i, 0u);
        let pred0 = mlp_xor_forward(pos, p0);
        let eps = 0.0001;
        let p0_clamped = clamp(pred0, eps, 1.0 - eps);
        total_loss = total_loss - log(1.0 - p0_clamped);  // Target = 0

        // Class 1 spiral
        let p1 = spiral_point(i, 1u);
        let pred1 = mlp_xor_forward(pos, p1);
        let p1_clamped = clamp(pred1, eps, 1.0 - eps);
        total_loss = total_loss - log(p1_clamped);  // Target = 1
    }

    return total_loss / f32(2u * SPIRAL_SIZE);
}

fn mlp_spiral_gradient(pos: array<f32, 256>, d_idx: u32) -> f32 {
    let eps = 0.001;
    var pos_plus = pos;
    var pos_minus = pos;
    pos_plus[d_idx] = pos[d_idx] + eps;
    pos_minus[d_idx] = pos[d_idx] - eps;
    return (mlp_spiral_loss(pos_plus) - mlp_spiral_loss(pos_minus)) / (2.0 * eps);
}

// ============================================================================
// DEEP MLP - 3 LAYER NETWORK ON CIRCLES DATASET
// ============================================================================
// Network: 2 inputs -> 4 hidden (tanh) -> 4 hidden (tanh) -> 1 output (sigmoid)
// Parameters (37 total):
//   pos[0..7]   = W1 (2x4 input->hidden1 weights)
//   pos[8..11]  = b1 (4 hidden1 biases)
//   pos[12..27] = W2 (4x4 hidden1->hidden2 weights)
//   pos[28..31] = b2 (4 hidden2 biases)
//   pos[32..35] = W3 (4x1 hidden2->output weights)
//   pos[36]     = b3 (1 output bias)
const DEEP_DIM: u32 = 37u;
const CIRCLES_SIZE: u32 = 50u;

// Generate points on two concentric circles (classification task)
fn circles_point(idx: u32, cls: u32) -> vec2<f32> {
    let theta = f32(idx) / f32(CIRCLES_SIZE) * 2.0 * PI;
    let r = select(0.5, 1.5, cls == 1u);  // Inner circle r=0.5, outer r=1.5
    // Add slight noise for realism
    let noise = f32(hash(idx + cls * 1000u) & 0xFFu) / 255.0 * 0.1 - 0.05;
    return vec2<f32>((r + noise) * cos(theta), (r + noise) * sin(theta));
}

fn mlp_deep_forward(pos: array<f32, 256>, input: vec2<f32>) -> f32 {
    // Layer 1: input (2) -> hidden1 (4) with tanh
    // W1 is stored as [w00, w01, w10, w11, w20, w21, w30, w31] (4 neurons x 2 inputs)
    let h1_0 = tanh(pos[0] * input.x + pos[1] * input.y + pos[8]);
    let h1_1 = tanh(pos[2] * input.x + pos[3] * input.y + pos[9]);
    let h1_2 = tanh(pos[4] * input.x + pos[5] * input.y + pos[10]);
    let h1_3 = tanh(pos[6] * input.x + pos[7] * input.y + pos[11]);

    // Layer 2: hidden1 (4) -> hidden2 (4) with tanh
    // W2 is 4x4 stored row-major at pos[12..27], b2 at pos[28..31]
    let h2_0 = tanh(pos[12] * h1_0 + pos[13] * h1_1 + pos[14] * h1_2 + pos[15] * h1_3 + pos[28]);
    let h2_1 = tanh(pos[16] * h1_0 + pos[17] * h1_1 + pos[18] * h1_2 + pos[19] * h1_3 + pos[29]);
    let h2_2 = tanh(pos[20] * h1_0 + pos[21] * h1_1 + pos[22] * h1_2 + pos[23] * h1_3 + pos[30]);
    let h2_3 = tanh(pos[24] * h1_0 + pos[25] * h1_1 + pos[26] * h1_2 + pos[27] * h1_3 + pos[31]);

    // Layer 3: hidden2 (4) -> output (1) with sigmoid
    // W3 at pos[32..35], b3 at pos[36]
    let out = sigmoid(pos[32] * h2_0 + pos[33] * h2_1 + pos[34] * h2_2 + pos[35] * h2_3 + pos[36]);
    return out;
}

fn mlp_deep_loss(pos: array<f32, 256>) -> f32 {
    var total_loss = 0.0;

    // Sample points from both circles
    for (var i = 0u; i < CIRCLES_SIZE; i = i + 1u) {
        // Inner circle (class 0)
        let p0 = circles_point(i, 0u);
        let pred0 = mlp_deep_forward(pos, p0);
        let eps = 0.0001;
        let p0_clamped = clamp(pred0, eps, 1.0 - eps);
        total_loss = total_loss - log(1.0 - p0_clamped);  // Target = 0

        // Outer circle (class 1)
        let p1 = circles_point(i, 1u);
        let pred1 = mlp_deep_forward(pos, p1);
        let p1_clamped = clamp(pred1, eps, 1.0 - eps);
        total_loss = total_loss - log(p1_clamped);  // Target = 1
    }

    return total_loss / f32(2u * CIRCLES_SIZE);
}

fn mlp_deep_gradient(pos: array<f32, 256>, d_idx: u32) -> f32 {
    // Only compute gradient for active parameters
    if d_idx >= DEEP_DIM {
        return 0.0;
    }
    let eps = 0.001;
    var pos_plus = pos;
    var pos_minus = pos;
    pos_plus[d_idx] = pos[d_idx] + eps;
    pos_minus[d_idx] = pos[d_idx] - eps;
    return (mlp_deep_loss(pos_plus) - mlp_deep_loss(pos_minus)) / (2.0 * eps);
}

// ============================================================================
// VECTORIZED GRADIENT COMPUTATION
// Computes full gradient vector at once to avoid O(D²) complexity
// ============================================================================

// Vectorized sphere gradient - O(D)
fn sphere_gradient_full(pos: array<f32, 256>, dim: u32, grad: ptr<function, array<f32, 256>>) {
    for (var i = 0u; i < dim; i = i + 1u) {
        (*grad)[i] = 2.0 * pos[i];
    }
}

// Vectorized Rosenbrock gradient - O(D)
fn rosenbrock_gradient_full(pos: array<f32, 256>, dim: u32, grad: ptr<function, array<f32, 256>>) {
    // Initialize all to zero
    for (var i = 0u; i < dim; i = i + 1u) {
        (*grad)[i] = 0.0;
    }

    for (var i = 0u; i < dim - 1u; i = i + 1u) {
        let x_i = pos[i];
        let x_i1 = pos[i + 1u];
        // Contribution to gradient[i] from term (i, i+1)
        (*grad)[i] = (*grad)[i] - 400.0 * x_i * (x_i1 - x_i * x_i) - 2.0 * (1.0 - x_i);
        // Contribution to gradient[i+1] from term (i, i+1)
        (*grad)[i + 1u] = (*grad)[i + 1u] + 200.0 * (x_i1 - x_i * x_i);
    }
}

// Vectorized Rastrigin gradient - O(D)
fn rastrigin_gradient_full(pos: array<f32, 256>, dim: u32, grad: ptr<function, array<f32, 256>>) {
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        (*grad)[i] = 2.0 * x + 20.0 * PI * fast_sin(TWO_PI * x);
    }
}

// Vectorized Ackley gradient - O(D) instead of O(D²)!
fn ackley_gradient_full(pos: array<f32, 256>, dim: u32, grad: ptr<function, array<f32, 256>>) {
    let a = 20.0;
    let b = 0.2;
    let n = f32(dim);

    // Compute sums ONCE - this was being done D times before!
    var sum_sq = 0.0;
    var sum_cos = 0.0;
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        sum_sq = sum_sq + x * x;
        sum_cos = sum_cos + fast_cos(TWO_PI * x);
    }

    // Precompute expensive terms
    let sqrt_term = fast_sqrt(sum_sq / n);
    let exp1 = fast_exp(-b * sqrt_term);
    let exp2 = fast_exp(sum_cos / n);

    // Compute coefficient for first term
    var coef1 = 0.0;
    if sqrt_term > 0.0001 {
        coef1 = a * b * exp1 / (n * sqrt_term);
    }

    // Compute all gradients using cached values - O(D) total
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        // Gradient of first term: coef1 * x_i
        // Gradient of second term: (2π/n) * sin(2π*x_i) * exp2
        (*grad)[i] = coef1 * x + TWO_PI * fast_sin(TWO_PI * x) / n * exp2;
    }
}

// Vectorized Schwefel gradient - O(D)
fn schwefel_gradient_full(pos: array<f32, 256>, dim: u32, grad: ptr<function, array<f32, 256>>) {
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        let abs_x = abs(x);

        if abs_x < 0.0001 {
            (*grad)[i] = 0.0;
        } else {
            let sqrt_abs_x = fast_sqrt(abs_x);
            let sign_x = select(-1.0, 1.0, x >= 0.0);
            let g = fast_sin(sqrt_abs_x) + sign_x * sqrt_abs_x * fast_cos(sqrt_abs_x) / 2.0;
            (*grad)[i] = -g;  // Negative because loss = constant - sum
        }
    }
}

// Vectorized Griewank gradient - O(D) with analytical gradient
fn griewank_gradient_full(pos: array<f32, 256>, dim: u32, grad: ptr<function, array<f32, 256>>) {
    // First compute the product term (needed for all gradients)
    var prod = 1.0;
    for (var i = 0u; i < dim; i = i + 1u) {
        prod = prod * fast_cos(pos[i] / fast_sqrt(f32(i + 1u)));
    }

    // Now compute each gradient component
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        let sqrt_i = fast_sqrt(f32(i + 1u));
        let cos_term = fast_cos(x / sqrt_i);

        // d/dx_i of sum term: x_i / 2000
        let sum_grad = x / 2000.0;

        // d/dx_i of product term: -sin(x_i/sqrt(i+1)) / sqrt(i+1) * (prod / cos(x_i/sqrt(i+1)))
        var prod_grad = 0.0;
        if abs(cos_term) > 0.0001 {
            prod_grad = fast_sin(x / sqrt_i) / sqrt_i * (prod / cos_term);
        }

        (*grad)[i] = sum_grad + prod_grad;
    }
}

// Vectorized Levy gradient - O(D) analytical
fn levy_gradient_full(pos: array<f32, 256>, dim: u32, grad: ptr<function, array<f32, 256>>) {
    // w_i = 1 + (x_i - 1) / 4
    // dw_i/dx_i = 1/4

    let w0 = 1.0 + (pos[0] - 1.0) / 4.0;
    let wn = 1.0 + (pos[dim - 1u] - 1.0) / 4.0;

    // Initialize gradients
    for (var i = 0u; i < dim; i = i + 1u) {
        (*grad)[i] = 0.0;
    }

    // Gradient of sin²(πw_0) term wrt x_0
    let sin_pw0 = fast_sin(PI * w0);
    let cos_pw0 = fast_cos(PI * w0);
    (*grad)[0] = (*grad)[0] + 2.0 * sin_pw0 * cos_pw0 * PI * 0.25;

    // Gradient of middle sum terms
    for (var i = 0u; i < dim - 1u; i = i + 1u) {
        let w = 1.0 + (pos[i] - 1.0) / 4.0;
        let w_next = 1.0 + (pos[i + 1u] - 1.0) / 4.0;
        let sin_term = fast_sin(PI * w_next);
        let cos_term = fast_cos(PI * w_next);

        // d/dx_i of (w-1)² * (1 + 10*sin²(πw_{i+1}))
        (*grad)[i] = (*grad)[i] + 2.0 * (w - 1.0) * 0.25 * (1.0 + 10.0 * sin_term * sin_term);

        // d/dx_{i+1} of (w-1)² * (1 + 10*sin²(πw_{i+1}))
        (*grad)[i + 1u] = (*grad)[i + 1u] + (w - 1.0) * (w - 1.0) * 20.0 * sin_term * cos_term * PI * 0.25;
    }

    // Gradient of final term: (w_n-1)² * (1 + sin²(2πw_n))
    let sin_2pwn = fast_sin(TWO_PI * wn);
    let cos_2pwn = fast_cos(TWO_PI * wn);
    (*grad)[dim - 1u] = (*grad)[dim - 1u] + 2.0 * (wn - 1.0) * 0.25 * (1.0 + sin_2pwn * sin_2pwn);
    (*grad)[dim - 1u] = (*grad)[dim - 1u] + (wn - 1.0) * (wn - 1.0) * 2.0 * sin_2pwn * cos_2pwn * TWO_PI * 0.25;
}

// Vectorized Styblinski-Tang gradient - O(D)
fn styblinski_tang_gradient_full(pos: array<f32, 256>, dim: u32, grad: ptr<function, array<f32, 256>>) {
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        (*grad)[i] = 2.0 * x * x * x - 16.0 * x + 2.5;
    }
}

// Vectorized multimodal gradient - O(D)
fn multimodal_gradient_full(pos: array<f32, 256>, dim: u32, grad: ptr<function, array<f32, 256>>) {
    for (var d = 0u; d < dim; d = d + 1u) {
        let pair_idx = d / 2u;
        let in_pair = d % 2u;
        let d_base = pair_idx * 2u;

        if d_base >= dim {
            (*grad)[d] = 0.0;
            continue;
        }

        let x = pos[d_base];
        let y = pos[min(d_base + 1u, dim - 1u)];

        let scale = 1.0 / (1.0 + f32(d_base) * 0.1);
        let target_x = 1.5 * scale;
        let target_y = 2.0 * scale;

        let d_pos = (x - target_x) * (x - target_x) + (y - target_y) * (y - target_y);
        let d_neg = (x + target_x) * (x + target_x) + (y + target_y) * (y + target_y);

        if d_pos < d_neg {
            if in_pair == 0u {
                (*grad)[d] = x - target_x;
            } else {
                (*grad)[d] = y - target_y;
            }
        } else {
            if in_pair == 0u {
                (*grad)[d] = x + target_x;
            } else {
                (*grad)[d] = y + target_y;
            }
        }
    }
}

// Vectorized 2D neural net gradient
fn nn_gradient_2d_full(pos: array<f32, 256>, dim: u32, grad: ptr<function, array<f32, 256>>) {
    let g = nn_gradient_2d(pos[0], pos[1]);
    (*grad)[0] = g.x;
    (*grad)[1] = g.y;
    for (var i = 2u; i < dim; i = i + 1u) {
        (*grad)[i] = 0.0;
    }
}

// Analytical MLP XOR gradient using backpropagation - O(1) loss evals instead of O(9)
// Network: 2->2->1 with tanh hidden and sigmoid output
// Parameters: W1[0..3], b1[4..5], W2[6..7], b2[8]
fn mlp_xor_gradient_full(pos: array<f32, 256>, dim: u32, grad: ptr<function, array<f32, 256>>) {
    // Initialize gradients to zero
    for (var i = 0u; i < 9u; i = i + 1u) {
        (*grad)[i] = 0.0;
    }
    for (var i = 9u; i < dim; i = i + 1u) {
        (*grad)[i] = 0.0;
    }

    // Accumulate gradients over all training examples
    for (var sample = 0u; sample < 4u; sample = sample + 1u) {
        let input = XOR_X[sample];
        let label = XOR_Y[sample];

        // Forward pass with intermediate values
        let z0 = pos[0] * input.x + pos[1] * input.y + pos[4];
        let z1 = pos[2] * input.x + pos[3] * input.y + pos[5];
        let h0 = tanh(z0);
        let h1 = tanh(z1);
        let z_out = pos[6] * h0 + pos[7] * h1 + pos[8];
        let pred = sigmoid(z_out);

        // Backward pass
        // dL/d(pred) for BCE with sigmoid = pred - label
        let d_out = pred - label;

        // Gradients for output layer (W2 and b2)
        (*grad)[6] = (*grad)[6] + d_out * h0;  // dL/dw6
        (*grad)[7] = (*grad)[7] + d_out * h1;  // dL/dw7
        (*grad)[8] = (*grad)[8] + d_out;       // dL/db2

        // Backprop through hidden layer (tanh derivative = 1 - tanh²)
        let d_h0 = d_out * pos[6] * (1.0 - h0 * h0);
        let d_h1 = d_out * pos[7] * (1.0 - h1 * h1);

        // Gradients for hidden layer (W1 and b1)
        (*grad)[0] = (*grad)[0] + d_h0 * input.x;  // dL/dw0
        (*grad)[1] = (*grad)[1] + d_h0 * input.y;  // dL/dw1
        (*grad)[2] = (*grad)[2] + d_h1 * input.x;  // dL/dw2
        (*grad)[3] = (*grad)[3] + d_h1 * input.y;  // dL/dw3
        (*grad)[4] = (*grad)[4] + d_h0;            // dL/db0
        (*grad)[5] = (*grad)[5] + d_h1;            // dL/db1
    }

    // Average over samples
    for (var i = 0u; i < 9u; i = i + 1u) {
        (*grad)[i] = (*grad)[i] / 4.0;
    }
}

// Analytical MLP spiral gradient using backpropagation
// Same network as XOR, different dataset
fn mlp_spiral_gradient_full(pos: array<f32, 256>, dim: u32, grad: ptr<function, array<f32, 256>>) {
    // Initialize gradients to zero
    for (var i = 0u; i < 9u; i = i + 1u) {
        (*grad)[i] = 0.0;
    }
    for (var i = 9u; i < dim; i = i + 1u) {
        (*grad)[i] = 0.0;
    }

    let n_samples = f32(2u * SPIRAL_SIZE);

    // Accumulate gradients over spiral points
    for (var i = 0u; i < SPIRAL_SIZE; i = i + 1u) {
        // Class 0 spiral (target = 0)
        let p0 = spiral_point(i, 0u);
        {
            let input = p0;
            let label = 0.0;

            let z0 = pos[0] * input.x + pos[1] * input.y + pos[4];
            let z1 = pos[2] * input.x + pos[3] * input.y + pos[5];
            let h0 = tanh(z0);
            let h1 = tanh(z1);
            let z_out = pos[6] * h0 + pos[7] * h1 + pos[8];
            let pred = sigmoid(z_out);

            let d_out = pred - label;

            (*grad)[6] = (*grad)[6] + d_out * h0;
            (*grad)[7] = (*grad)[7] + d_out * h1;
            (*grad)[8] = (*grad)[8] + d_out;

            let d_h0 = d_out * pos[6] * (1.0 - h0 * h0);
            let d_h1 = d_out * pos[7] * (1.0 - h1 * h1);

            (*grad)[0] = (*grad)[0] + d_h0 * input.x;
            (*grad)[1] = (*grad)[1] + d_h0 * input.y;
            (*grad)[2] = (*grad)[2] + d_h1 * input.x;
            (*grad)[3] = (*grad)[3] + d_h1 * input.y;
            (*grad)[4] = (*grad)[4] + d_h0;
            (*grad)[5] = (*grad)[5] + d_h1;
        }

        // Class 1 spiral (target = 1)
        let p1 = spiral_point(i, 1u);
        {
            let input = p1;
            let label = 1.0;

            let z0 = pos[0] * input.x + pos[1] * input.y + pos[4];
            let z1 = pos[2] * input.x + pos[3] * input.y + pos[5];
            let h0 = tanh(z0);
            let h1 = tanh(z1);
            let z_out = pos[6] * h0 + pos[7] * h1 + pos[8];
            let pred = sigmoid(z_out);

            let d_out = pred - label;

            (*grad)[6] = (*grad)[6] + d_out * h0;
            (*grad)[7] = (*grad)[7] + d_out * h1;
            (*grad)[8] = (*grad)[8] + d_out;

            let d_h0 = d_out * pos[6] * (1.0 - h0 * h0);
            let d_h1 = d_out * pos[7] * (1.0 - h1 * h1);

            (*grad)[0] = (*grad)[0] + d_h0 * input.x;
            (*grad)[1] = (*grad)[1] + d_h0 * input.y;
            (*grad)[2] = (*grad)[2] + d_h1 * input.x;
            (*grad)[3] = (*grad)[3] + d_h1 * input.y;
            (*grad)[4] = (*grad)[4] + d_h0;
            (*grad)[5] = (*grad)[5] + d_h1;
        }
    }

    // Average over samples
    for (var i = 0u; i < 9u; i = i + 1u) {
        (*grad)[i] = (*grad)[i] / n_samples;
    }
}

// Analytical deep MLP gradient using backpropagation - O(1) loss evals instead of O(37)
// Network: 2->4->4->1 with tanh hidden layers and sigmoid output
// Parameters: W1[0..7], b1[8..11], W2[12..27], b2[28..31], W3[32..35], b3[36]
fn mlp_deep_gradient_full(pos: array<f32, 256>, dim: u32, grad: ptr<function, array<f32, 256>>) {
    // Initialize all gradients to zero
    for (var i = 0u; i < DEEP_DIM; i = i + 1u) {
        (*grad)[i] = 0.0;
    }
    for (var i = DEEP_DIM; i < dim; i = i + 1u) {
        (*grad)[i] = 0.0;
    }

    let n_samples = f32(2u * CIRCLES_SIZE);

    // Accumulate gradients over all circle points
    for (var sample = 0u; sample < CIRCLES_SIZE; sample = sample + 1u) {
        // Inner circle (class 0)
        {
            let input = circles_point(sample, 0u);
            let label = 0.0;

            // Forward pass - Layer 1
            let z1_0 = pos[0] * input.x + pos[1] * input.y + pos[8];
            let z1_1 = pos[2] * input.x + pos[3] * input.y + pos[9];
            let z1_2 = pos[4] * input.x + pos[5] * input.y + pos[10];
            let z1_3 = pos[6] * input.x + pos[7] * input.y + pos[11];
            let h1_0 = tanh(z1_0);
            let h1_1 = tanh(z1_1);
            let h1_2 = tanh(z1_2);
            let h1_3 = tanh(z1_3);

            // Forward pass - Layer 2
            let z2_0 = pos[12] * h1_0 + pos[13] * h1_1 + pos[14] * h1_2 + pos[15] * h1_3 + pos[28];
            let z2_1 = pos[16] * h1_0 + pos[17] * h1_1 + pos[18] * h1_2 + pos[19] * h1_3 + pos[29];
            let z2_2 = pos[20] * h1_0 + pos[21] * h1_1 + pos[22] * h1_2 + pos[23] * h1_3 + pos[30];
            let z2_3 = pos[24] * h1_0 + pos[25] * h1_1 + pos[26] * h1_2 + pos[27] * h1_3 + pos[31];
            let h2_0 = tanh(z2_0);
            let h2_1 = tanh(z2_1);
            let h2_2 = tanh(z2_2);
            let h2_3 = tanh(z2_3);

            // Forward pass - Output layer
            let z_out = pos[32] * h2_0 + pos[33] * h2_1 + pos[34] * h2_2 + pos[35] * h2_3 + pos[36];
            let pred = sigmoid(z_out);

            // Backward pass
            let d_out = pred - label;

            // Output layer gradients
            (*grad)[32] = (*grad)[32] + d_out * h2_0;
            (*grad)[33] = (*grad)[33] + d_out * h2_1;
            (*grad)[34] = (*grad)[34] + d_out * h2_2;
            (*grad)[35] = (*grad)[35] + d_out * h2_3;
            (*grad)[36] = (*grad)[36] + d_out;

            // Backprop to layer 2
            let d_h2_0 = d_out * pos[32] * (1.0 - h2_0 * h2_0);
            let d_h2_1 = d_out * pos[33] * (1.0 - h2_1 * h2_1);
            let d_h2_2 = d_out * pos[34] * (1.0 - h2_2 * h2_2);
            let d_h2_3 = d_out * pos[35] * (1.0 - h2_3 * h2_3);

            // Layer 2 weight gradients
            (*grad)[12] = (*grad)[12] + d_h2_0 * h1_0;
            (*grad)[13] = (*grad)[13] + d_h2_0 * h1_1;
            (*grad)[14] = (*grad)[14] + d_h2_0 * h1_2;
            (*grad)[15] = (*grad)[15] + d_h2_0 * h1_3;
            (*grad)[16] = (*grad)[16] + d_h2_1 * h1_0;
            (*grad)[17] = (*grad)[17] + d_h2_1 * h1_1;
            (*grad)[18] = (*grad)[18] + d_h2_1 * h1_2;
            (*grad)[19] = (*grad)[19] + d_h2_1 * h1_3;
            (*grad)[20] = (*grad)[20] + d_h2_2 * h1_0;
            (*grad)[21] = (*grad)[21] + d_h2_2 * h1_1;
            (*grad)[22] = (*grad)[22] + d_h2_2 * h1_2;
            (*grad)[23] = (*grad)[23] + d_h2_2 * h1_3;
            (*grad)[24] = (*grad)[24] + d_h2_3 * h1_0;
            (*grad)[25] = (*grad)[25] + d_h2_3 * h1_1;
            (*grad)[26] = (*grad)[26] + d_h2_3 * h1_2;
            (*grad)[27] = (*grad)[27] + d_h2_3 * h1_3;
            (*grad)[28] = (*grad)[28] + d_h2_0;
            (*grad)[29] = (*grad)[29] + d_h2_1;
            (*grad)[30] = (*grad)[30] + d_h2_2;
            (*grad)[31] = (*grad)[31] + d_h2_3;

            // Backprop to layer 1
            let d_h1_0 = (d_h2_0 * pos[12] + d_h2_1 * pos[16] + d_h2_2 * pos[20] + d_h2_3 * pos[24]) * (1.0 - h1_0 * h1_0);
            let d_h1_1 = (d_h2_0 * pos[13] + d_h2_1 * pos[17] + d_h2_2 * pos[21] + d_h2_3 * pos[25]) * (1.0 - h1_1 * h1_1);
            let d_h1_2 = (d_h2_0 * pos[14] + d_h2_1 * pos[18] + d_h2_2 * pos[22] + d_h2_3 * pos[26]) * (1.0 - h1_2 * h1_2);
            let d_h1_3 = (d_h2_0 * pos[15] + d_h2_1 * pos[19] + d_h2_2 * pos[23] + d_h2_3 * pos[27]) * (1.0 - h1_3 * h1_3);

            // Layer 1 weight gradients
            (*grad)[0] = (*grad)[0] + d_h1_0 * input.x;
            (*grad)[1] = (*grad)[1] + d_h1_0 * input.y;
            (*grad)[2] = (*grad)[2] + d_h1_1 * input.x;
            (*grad)[3] = (*grad)[3] + d_h1_1 * input.y;
            (*grad)[4] = (*grad)[4] + d_h1_2 * input.x;
            (*grad)[5] = (*grad)[5] + d_h1_2 * input.y;
            (*grad)[6] = (*grad)[6] + d_h1_3 * input.x;
            (*grad)[7] = (*grad)[7] + d_h1_3 * input.y;
            (*grad)[8] = (*grad)[8] + d_h1_0;
            (*grad)[9] = (*grad)[9] + d_h1_1;
            (*grad)[10] = (*grad)[10] + d_h1_2;
            (*grad)[11] = (*grad)[11] + d_h1_3;
        }

        // Outer circle (class 1)
        {
            let input = circles_point(sample, 1u);
            let label = 1.0;

            // Forward pass - Layer 1
            let z1_0 = pos[0] * input.x + pos[1] * input.y + pos[8];
            let z1_1 = pos[2] * input.x + pos[3] * input.y + pos[9];
            let z1_2 = pos[4] * input.x + pos[5] * input.y + pos[10];
            let z1_3 = pos[6] * input.x + pos[7] * input.y + pos[11];
            let h1_0 = tanh(z1_0);
            let h1_1 = tanh(z1_1);
            let h1_2 = tanh(z1_2);
            let h1_3 = tanh(z1_3);

            // Forward pass - Layer 2
            let z2_0 = pos[12] * h1_0 + pos[13] * h1_1 + pos[14] * h1_2 + pos[15] * h1_3 + pos[28];
            let z2_1 = pos[16] * h1_0 + pos[17] * h1_1 + pos[18] * h1_2 + pos[19] * h1_3 + pos[29];
            let z2_2 = pos[20] * h1_0 + pos[21] * h1_1 + pos[22] * h1_2 + pos[23] * h1_3 + pos[30];
            let z2_3 = pos[24] * h1_0 + pos[25] * h1_1 + pos[26] * h1_2 + pos[27] * h1_3 + pos[31];
            let h2_0 = tanh(z2_0);
            let h2_1 = tanh(z2_1);
            let h2_2 = tanh(z2_2);
            let h2_3 = tanh(z2_3);

            // Forward pass - Output layer
            let z_out = pos[32] * h2_0 + pos[33] * h2_1 + pos[34] * h2_2 + pos[35] * h2_3 + pos[36];
            let pred = sigmoid(z_out);

            // Backward pass
            let d_out = pred - label;

            // Output layer gradients
            (*grad)[32] = (*grad)[32] + d_out * h2_0;
            (*grad)[33] = (*grad)[33] + d_out * h2_1;
            (*grad)[34] = (*grad)[34] + d_out * h2_2;
            (*grad)[35] = (*grad)[35] + d_out * h2_3;
            (*grad)[36] = (*grad)[36] + d_out;

            // Backprop to layer 2
            let d_h2_0 = d_out * pos[32] * (1.0 - h2_0 * h2_0);
            let d_h2_1 = d_out * pos[33] * (1.0 - h2_1 * h2_1);
            let d_h2_2 = d_out * pos[34] * (1.0 - h2_2 * h2_2);
            let d_h2_3 = d_out * pos[35] * (1.0 - h2_3 * h2_3);

            // Layer 2 weight gradients
            (*grad)[12] = (*grad)[12] + d_h2_0 * h1_0;
            (*grad)[13] = (*grad)[13] + d_h2_0 * h1_1;
            (*grad)[14] = (*grad)[14] + d_h2_0 * h1_2;
            (*grad)[15] = (*grad)[15] + d_h2_0 * h1_3;
            (*grad)[16] = (*grad)[16] + d_h2_1 * h1_0;
            (*grad)[17] = (*grad)[17] + d_h2_1 * h1_1;
            (*grad)[18] = (*grad)[18] + d_h2_1 * h1_2;
            (*grad)[19] = (*grad)[19] + d_h2_1 * h1_3;
            (*grad)[20] = (*grad)[20] + d_h2_2 * h1_0;
            (*grad)[21] = (*grad)[21] + d_h2_2 * h1_1;
            (*grad)[22] = (*grad)[22] + d_h2_2 * h1_2;
            (*grad)[23] = (*grad)[23] + d_h2_2 * h1_3;
            (*grad)[24] = (*grad)[24] + d_h2_3 * h1_0;
            (*grad)[25] = (*grad)[25] + d_h2_3 * h1_1;
            (*grad)[26] = (*grad)[26] + d_h2_3 * h1_2;
            (*grad)[27] = (*grad)[27] + d_h2_3 * h1_3;
            (*grad)[28] = (*grad)[28] + d_h2_0;
            (*grad)[29] = (*grad)[29] + d_h2_1;
            (*grad)[30] = (*grad)[30] + d_h2_2;
            (*grad)[31] = (*grad)[31] + d_h2_3;

            // Backprop to layer 1
            let d_h1_0 = (d_h2_0 * pos[12] + d_h2_1 * pos[16] + d_h2_2 * pos[20] + d_h2_3 * pos[24]) * (1.0 - h1_0 * h1_0);
            let d_h1_1 = (d_h2_0 * pos[13] + d_h2_1 * pos[17] + d_h2_2 * pos[21] + d_h2_3 * pos[25]) * (1.0 - h1_1 * h1_1);
            let d_h1_2 = (d_h2_0 * pos[14] + d_h2_1 * pos[18] + d_h2_2 * pos[22] + d_h2_3 * pos[26]) * (1.0 - h1_2 * h1_2);
            let d_h1_3 = (d_h2_0 * pos[15] + d_h2_1 * pos[19] + d_h2_2 * pos[23] + d_h2_3 * pos[27]) * (1.0 - h1_3 * h1_3);

            // Layer 1 weight gradients
            (*grad)[0] = (*grad)[0] + d_h1_0 * input.x;
            (*grad)[1] = (*grad)[1] + d_h1_0 * input.y;
            (*grad)[2] = (*grad)[2] + d_h1_1 * input.x;
            (*grad)[3] = (*grad)[3] + d_h1_1 * input.y;
            (*grad)[4] = (*grad)[4] + d_h1_2 * input.x;
            (*grad)[5] = (*grad)[5] + d_h1_2 * input.y;
            (*grad)[6] = (*grad)[6] + d_h1_3 * input.x;
            (*grad)[7] = (*grad)[7] + d_h1_3 * input.y;
            (*grad)[8] = (*grad)[8] + d_h1_0;
            (*grad)[9] = (*grad)[9] + d_h1_1;
            (*grad)[10] = (*grad)[10] + d_h1_2;
            (*grad)[11] = (*grad)[11] + d_h1_3;
        }
    }

    // Average over samples
    for (var i = 0u; i < DEEP_DIM; i = i + 1u) {
        (*grad)[i] = (*grad)[i] / n_samples;
    }
}

// Vectorized custom gradient - calls user-provided custom_gradient for each component
fn custom_gradient_full(pos: array<f32, 256>, dim: u32, grad: ptr<function, array<f32, 256>>) {
    for (var i = 0u; i < dim; i = i + 1u) {
        (*grad)[i] = custom_gradient(pos, dim, i);
    }
}

// Master dispatch for vectorized gradients
fn compute_gradient_full(pos: array<f32, 256>, dim: u32, loss_fn: u32, grad: ptr<function, array<f32, 256>>) {
    switch loss_fn {
        case 0u: { nn_gradient_2d_full(pos, dim, grad); }
        case 1u: { multimodal_gradient_full(pos, dim, grad); }
        case 2u: { rosenbrock_gradient_full(pos, dim, grad); }
        case 3u: { rastrigin_gradient_full(pos, dim, grad); }
        case 4u: { ackley_gradient_full(pos, dim, grad); }
        case 5u: { sphere_gradient_full(pos, dim, grad); }
        case 6u: { mlp_xor_gradient_full(pos, dim, grad); }
        case 7u: { mlp_spiral_gradient_full(pos, dim, grad); }
        case 8u: { mlp_deep_gradient_full(pos, dim, grad); }
        case 9u: { schwefel_gradient_full(pos, dim, grad); }
        case 10u: { custom_gradient_full(pos, dim, grad); }
        case 11u: { griewank_gradient_full(pos, dim, grad); }
        case 12u: { levy_gradient_full(pos, dim, grad); }
        case 13u: { styblinski_tang_gradient_full(pos, dim, grad); }
        default: { sphere_gradient_full(pos, dim, grad); }
    }
}

// ============================================================================
// CLASSIC OPTIMIZATION BENCHMARK FUNCTIONS
// ============================================================================

// Rosenbrock function (N-dimensional)
// Global minimum: f(1,1,...,1) = 0
// Famous "banana valley" - tests ability to follow narrow curved valleys
fn rosenbrock_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < dim - 1u; i = i + 1u) {
        let x_i = pos[i];
        let x_i1 = pos[i + 1u];
        sum = sum + 100.0 * (x_i1 - x_i * x_i) * (x_i1 - x_i * x_i) + (1.0 - x_i) * (1.0 - x_i);
    }
    return sum;
}

fn rosenbrock_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    var grad = 0.0;
    let x_i = pos[d_idx];

    // Contribution from term (i, i+1) where this is x_i
    if d_idx < dim - 1u {
        let x_i1 = pos[d_idx + 1u];
        grad = grad - 400.0 * x_i * (x_i1 - x_i * x_i) - 2.0 * (1.0 - x_i);
    }

    // Contribution from term (i-1, i) where this is x_{i+1}
    if d_idx > 0u {
        let x_im1 = pos[d_idx - 1u];
        grad = grad + 200.0 * (x_i - x_im1 * x_im1);
    }

    return grad;
}

// Rastrigin function (N-dimensional)
// Global minimum: f(0,0,...,0) = 0
// Highly multimodal with regular grid of local minima - tests global search
fn rastrigin_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    var sum = 10.0 * f32(dim);
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        sum = sum + x * x - 10.0 * fast_cos(TWO_PI * x);
    }
    return sum;
}

fn rastrigin_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    let x = pos[d_idx];
    return 2.0 * x + 20.0 * PI * fast_sin(TWO_PI * x);
}

// Ackley function (N-dimensional)
// Global minimum: f(0,0,...,0) = 0
// Nearly flat outer region with deep hole at center - tests exploitation
fn ackley_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    let a = 20.0;
    let b = 0.2;

    var sum_sq = 0.0;
    var sum_cos = 0.0;
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        sum_sq = sum_sq + x * x;
        sum_cos = sum_cos + fast_cos(TWO_PI * x);
    }

    let n = f32(dim);
    let sqrt_term = fast_sqrt(sum_sq / n);
    return -a * fast_exp(-b * sqrt_term) - fast_exp(sum_cos / n) + a + 2.71828182845;
}

fn ackley_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    let a = 20.0;
    let b = 0.2;

    var sum_sq = 0.0;
    var sum_cos = 0.0;
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        sum_sq = sum_sq + x * x;
        sum_cos = sum_cos + fast_cos(TWO_PI * x);
    }

    let n = f32(dim);
    let x_d = pos[d_idx];
    let sqrt_term = fast_sqrt(sum_sq / n);

    // Gradient of first term: a * b * x_d / (n * sqrt_term) * exp(-b * sqrt_term)
    var grad = 0.0;
    if sqrt_term > 0.0001 {
        grad = a * b * x_d / (n * sqrt_term) * fast_exp(-b * sqrt_term);
    }

    // Gradient of second term: c * sin(c * x_d) / n * exp(sum_cos / n)
    grad = grad + TWO_PI * fast_sin(TWO_PI * x_d) / n * fast_exp(sum_cos / n);

    return grad;
}

// Sphere function (N-dimensional) - simple convex baseline
// Global minimum: f(0,0,...,0) = 0
fn sphere_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < dim; i = i + 1u) {
        sum = sum + pos[i] * pos[i];
    }
    return sum;
}

fn sphere_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    return 2.0 * pos[d_idx];
}

// Schwefel function (N-dimensional) - deceptive global minimum far from origin
// Global minimum: f(420.9687, ..., 420.9687) = 0
// The global minimum is far from the origin, and local minima are deceptive
fn schwefel_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        sum = sum + x * fast_sin(fast_sqrt(abs(x)));
    }
    return 418.9829 * f32(dim) - sum;
}

fn schwefel_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    let x = pos[d_idx];
    let abs_x = abs(x);

    if abs_x < 0.0001 {
        return 0.0;
    }

    let sqrt_abs_x = fast_sqrt(abs_x);
    let sign_x = select(-1.0, 1.0, x >= 0.0);

    // d/dx [x * sin(sqrt(|x|))] = sin(sqrt(|x|)) + x * cos(sqrt(|x|)) * (sign(x) / (2 * sqrt(|x|)))
    // = sin(sqrt(|x|)) + sign(x) * sqrt(|x|) * cos(sqrt(|x|)) / 2
    let grad = fast_sin(sqrt_abs_x) + sign_x * sqrt_abs_x * fast_cos(sqrt_abs_x) / 2.0;

    // Negative because schwefel_loss = 418.9829*n - sum
    return -grad;
}

// Griewank function (N-dimensional)
// Global minimum: f(0, 0, ..., 0) = 0
// f(x) = 1 + sum(x_i^2 / 4000) - prod(cos(x_i / sqrt(i+1)))
fn griewank_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    var sum_term = 0.0;
    var prod_term = 1.0;
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        sum_term = sum_term + x * x / 4000.0;
        prod_term = prod_term * fast_cos(x / fast_sqrt(f32(i + 1u)));
    }
    return 1.0 + sum_term - prod_term;
}

fn griewank_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    // Numerical gradient for product term complexity
    let eps = 0.001;
    var pos_plus = pos;
    var pos_minus = pos;
    pos_plus[d_idx] = pos[d_idx] + eps;
    pos_minus[d_idx] = pos[d_idx] - eps;
    return (griewank_loss(pos_plus, dim) - griewank_loss(pos_minus, dim)) / (2.0 * eps);
}

// Levy function (N-dimensional)
// Global minimum: f(1, 1, ..., 1) = 0
// w_i = 1 + (x_i - 1) / 4
// f(x) = sin²(πw_1) + sum[(w_i-1)²(1+10sin²(πw_i+1))] + (w_n-1)²(1+sin²(2πw_n))
fn levy_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    let w0 = 1.0 + (pos[0] - 1.0) / 4.0;
    let wn = 1.0 + (pos[dim - 1u] - 1.0) / 4.0;

    let sin_pw0 = fast_sin(PI * w0);
    var sum = sin_pw0 * sin_pw0;

    for (var i = 0u; i < dim - 1u; i = i + 1u) {
        let w = 1.0 + (pos[i] - 1.0) / 4.0;
        let w_next = 1.0 + (pos[i + 1u] - 1.0) / 4.0;
        let sin_term = fast_sin(PI * w_next);
        sum = sum + (w - 1.0) * (w - 1.0) * (1.0 + 10.0 * sin_term * sin_term);
    }

    let sin_2pwn = fast_sin(TWO_PI * wn);
    sum = sum + (wn - 1.0) * (wn - 1.0) * (1.0 + sin_2pwn * sin_2pwn);

    return sum;
}

fn levy_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    let eps = 0.001;
    var pos_plus = pos;
    var pos_minus = pos;
    pos_plus[d_idx] = pos[d_idx] + eps;
    pos_minus[d_idx] = pos[d_idx] - eps;
    return (levy_loss(pos_plus, dim) - levy_loss(pos_minus, dim)) / (2.0 * eps);
}

// Styblinski-Tang function (N-dimensional)
// Global minimum: f(-2.903534, ..., -2.903534) ≈ -39.16599 * n
// f(x) = 0.5 * sum(x_i^4 - 16*x_i^2 + 5*x_i)
fn styblinski_tang_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        let x2 = x * x;
        sum = sum + x2 * x2 - 16.0 * x2 + 5.0 * x;
    }
    return 0.5 * sum;
}

fn styblinski_tang_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    let x = pos[d_idx];
    // d/dx [0.5 * (x^4 - 16x^2 + 5x)] = 0.5 * (4x^3 - 32x + 5) = 2x^3 - 16x + 2.5
    return 2.0 * x * x * x - 16.0 * x + 2.5;
}

// ============================================================================
// HAMILTONIAN MONTE CARLO (HMC) IMPLEMENTATION
//
// HMC uses Hamiltonian dynamics for efficient exploration:
//   H(q,p) = U(q) + K(p) where U = potential (loss), K = kinetic (|p|²/2m)
//
// Temperature controls momentum distribution: p ~ N(0, mT)
//   T → 0:   Small momenta → momentum-based optimization
//   T ~ 0.1: Balanced → efficient posterior sampling
//   T >> 1:  Large momenta → chaotic exploration (entropy)
// ============================================================================

// Helper: Compute loss for any loss function
fn compute_loss(pos: array<f32, 256>, dim: u32, loss_fn: u32) -> f32 {
    switch loss_fn {
        case 0u: { return nn_loss_2d(pos[0], pos[1]); }
        case 1u: { return multimodal_loss_nd(pos, dim); }
        case 2u: { return rosenbrock_loss(pos, dim); }
        case 3u: { return rastrigin_loss(pos, dim); }
        case 4u: { return ackley_loss(pos, dim); }
        case 5u: { return sphere_loss(pos, dim); }
        case 6u: { return mlp_xor_loss(pos); }
        case 7u: { return mlp_spiral_loss(pos); }
        case 8u: { return mlp_deep_loss(pos); }
        case 9u: { return schwefel_loss(pos, dim); }
        case 10u: { return custom_loss(pos, dim); }
        case 11u: { return griewank_loss(pos, dim); }
        case 12u: { return levy_loss(pos, dim); }
        case 13u: { return styblinski_tang_loss(pos, dim); }
        default: { return sphere_loss(pos, dim); }
    }
}

// Helper: Compute gradient for any loss function
fn compute_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32, loss_fn: u32) -> f32 {
    switch loss_fn {
        case 0u: {
            let grad = nn_gradient_2d(pos[0], pos[1]);
            if d_idx == 0u { return grad.x; }
            else if d_idx == 1u { return grad.y; }
            else { return 0.0; }
        }
        case 1u: { return multimodal_gradient_nd(pos, dim, d_idx); }
        case 2u: { return rosenbrock_gradient(pos, dim, d_idx); }
        case 3u: { return rastrigin_gradient(pos, dim, d_idx); }
        case 4u: { return ackley_gradient(pos, dim, d_idx); }
        case 5u: { return sphere_gradient(pos, dim, d_idx); }
        case 6u: { return mlp_xor_gradient(pos, d_idx); }
        case 7u: { return mlp_spiral_gradient(pos, d_idx); }
        case 8u: { return mlp_deep_gradient(pos, d_idx); }
        case 9u: { return schwefel_gradient(pos, dim, d_idx); }
        case 10u: { return custom_gradient(pos, dim, d_idx); }
        case 11u: { return griewank_gradient(pos, dim, d_idx); }
        case 12u: { return levy_gradient(pos, dim, d_idx); }
        case 13u: { return styblinski_tang_gradient(pos, dim, d_idx); }
        default: { return 2.0 * pos[d_idx]; }  // sphere gradient
    }
}

// Helper: Compute kinetic energy K = |p|²/2m
fn compute_kinetic_energy(vel: array<f32, 256>, dim: u32, mass: f32) -> f32 {
    var ke = 0.0;
    for (var d = 0u; d < dim; d = d + 1u) {
        ke = ke + vel[d] * vel[d];
    }
    return ke / (2.0 * mass);
}

// ============================================================================
// PARTICLE INITIALIZATION
// Initialize particles with random positions in [pos_min, pos_max]
// Much faster than CPU initialization for large particle counts
// ============================================================================
@compute @workgroup_size(256)
fn initialize_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let pc = uniforms.particle_count;
    if idx >= pc { return; }

    // Initialize position with uniform random in [pos_min, pos_max]
    // Using SoA dimension-major layout for coalesced access
    let range = uniforms.pos_max - uniforms.pos_min;
    for (var d = 0u; d < uniforms.dim; d = d + 1u) {
        // XOR-shift random number generator (same as CPU version)
        var seed = uniforms.seed + idx * 1000u + d;
        seed = seed ^ (seed << 13u);
        seed = seed ^ (seed >> 7u);
        seed = seed ^ (seed << 17u);
        let rand_val = f32(seed & 0xFFFFu) / 65535.0;
        // SoA write: positions[dim * particle_count + particle_idx]
        positions[d * pc + idx] = f16(uniforms.pos_min + rand_val * range);
        velocities[d * pc + idx] = f16(0.0);
    }

    // Initialize scalar fields
    var s: ParticleScalars;
    s.energy = 0.0;
    s.kinetic_energy = 0.0;
    s.mass = uniforms.mass;
    s.entropy_bits = 0u;
    scalars[idx] = s;
}

// ============================================================================
// HMC Pass 1: REFRESH MOMENTUM
// Sample p ~ N(0, mT) from Maxwell-Boltzmann distribution
// Store initial state for later accept/reject decision
// ============================================================================
@compute @workgroup_size(256)
fn refresh_momentum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let pc = uniforms.particle_count;
    if idx >= pc { return; }

    let m = uniforms.mass;
    let T = uniforms.temperature;
    let sigma = sqrt(m * T);  // Standard deviation of Maxwell-Boltzmann
    let dim = uniforms.dim;

    // Load positions into local array and sample new momentum (SoA access)
    var pos_f32: array<f32, 256>;
    var vel_f32: array<f32, 256>;
    for (var d = 0u; d < dim; d = d + 1u) {
        // Read position from SoA
        pos_f32[d] = f32(positions[d * pc + idx]);

        // Sample momentum from N(0, σ²)
        let seed1 = uniforms.seed + idx * 17u + d * 31u + 99999u;
        let seed2 = uniforms.seed + idx * 37u + d * 53u + 77777u;
        vel_f32[d] = sigma * randn(seed1, seed2);

        // Write new velocity to SoA
        velocities[d * pc + idx] = f16(vel_f32[d]);

        // Store initial state in proposal buffers (SoA)
        proposal_pos[d * pc + idx] = f16(pos_f32[d]);
        proposal_vel[d * pc + idx] = f16(vel_f32[d]);
    }

    // Compute initial potential energy U(q)
    let energy = compute_loss(pos_f32, dim, uniforms.loss_fn);

    // Compute initial kinetic energy K = |p|²/2m
    let kinetic_energy = compute_kinetic_energy(vel_f32, dim, m);

    // Update scalar fields
    var s: ParticleScalars;
    s.energy = energy;
    s.kinetic_energy = kinetic_energy;
    s.mass = m;
    s.entropy_bits = scalars[idx].entropy_bits;
    scalars[idx] = s;

    // Store initial scalars in proposal
    proposal_scalars[idx] = s;
}

// ============================================================================
// HMC Pass 2: BATCHED LEAPFROG INTEGRATION
// Symplectic integrator that preserves phase space volume
// All L leapfrog steps run in a SINGLE dispatch (no global memory round-trips)
//
// Leapfrog scheme (per step):
//   p(t + ε/2) = p(t) - (ε/2)∇U(q(t))
//   q(t + ε)   = q(t) + ε·p(t + ε/2)/m
//   p(t + ε)   = p(t + ε/2) - (ε/2)∇U(q(t + ε))
//
// Key optimization: Uses vectorized gradients (O(D) instead of O(D²) for Ackley etc.)
// ============================================================================
@compute @workgroup_size(256)
fn leapfrog_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let pc = uniforms.particle_count;
    if idx >= pc { return; }

    let eps = uniforms.step_size;
    let m = uniforms.mass;
    let L = uniforms.leapfrog_steps;
    let dim = uniforms.dim;

    // Load into local registers from SoA buffers (coalesced reads)
    var pos_f32: array<f32, 256>;
    var vel_f32: array<f32, 256>;
    for (var d = 0u; d < dim; d = d + 1u) {
        pos_f32[d] = f32(positions[d * pc + idx]);
        vel_f32[d] = f32(velocities[d * pc + idx]);
    }

    // Gradient array (reused across all L steps)
    var grad: array<f32, 256>;

    // Position domain bounds from uniforms (configurable per loss function)
    let pos_min = uniforms.pos_min;
    let pos_max = uniforms.pos_max;

    // Run ALL L leapfrog steps in local registers
    for (var l = 0u; l < L; l = l + 1u) {
        // Compute full gradient vector ONCE (vectorized - O(D) not O(D²))
        compute_gradient_full(pos_f32, dim, uniforms.loss_fn, &grad);

        // Half step for momentum: p(t + ε/2) = p(t) - (ε/2)∇U(q(t))
        for (var d = 0u; d < dim; d = d + 1u) {
            let grad_clipped = clamp(grad[d], -10.0, 10.0);
            vel_f32[d] = vel_f32[d] - 0.5 * eps * grad_clipped;
        }

        // Full step for position: q(t + ε) = q(t) + ε·p(t + ε/2)/m
        for (var d = 0u; d < dim; d = d + 1u) {
            pos_f32[d] = clamp(pos_f32[d] + eps * vel_f32[d] / m, pos_min, pos_max);
        }

        // Compute gradient at new position for second half-step
        compute_gradient_full(pos_f32, dim, uniforms.loss_fn, &grad);

        // Half step for momentum: p(t + ε) = p(t + ε/2) - (ε/2)∇U(q(t + ε))
        for (var d = 0u; d < dim; d = d + 1u) {
            let grad_clipped = clamp(grad[d], -10.0, 10.0);
            vel_f32[d] = vel_f32[d] - 0.5 * eps * grad_clipped;
        }
    }

    // Compute final energy
    let energy = compute_loss(pos_f32, dim, uniforms.loss_fn);

    // Write back to SoA buffers (coalesced writes)
    for (var d = 0u; d < dim; d = d + 1u) {
        positions[d * pc + idx] = f16(pos_f32[d]);
        velocities[d * pc + idx] = f16(vel_f32[d]);
    }

    // Update scalar fields
    var s = scalars[idx];
    s.energy = energy;
    s.kinetic_energy = compute_kinetic_energy(vel_f32, dim, m);
    scalars[idx] = s;
}

// ============================================================================
// HMC Pass 3: METROPOLIS ACCEPT/REJECT
// Accept proposal with probability min(1, exp(-ΔH))
// where ΔH = H_final - H_initial = (U_f + K_f) - (U_i + K_i)
// ============================================================================
@compute @workgroup_size(256)
fn accept_reject(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let pc = uniforms.particle_count;
    if idx >= pc { return; }

    let dim = uniforms.dim;

    // Read initial (proposal) and current scalars
    let initial_scalars = proposal_scalars[idx];
    let current_scalars = scalars[idx];

    // Compute Hamiltonians: H = U + K
    let H_initial = initial_scalars.energy + initial_scalars.kinetic_energy;
    let H_current = current_scalars.energy + current_scalars.kinetic_energy;
    let delta_H = H_current - H_initial;

    // Accept probability: min(1, exp(-ΔH/T))
    // Note: At T=1, this is standard HMC. At other T, we scale appropriately.
    let accept_prob = min(1.0, exp(-delta_H / max(uniforms.temperature, 0.001)));

    // Random number for accept/reject decision
    let u = rand(uniforms.seed + idx * 9973u + 123456u);

    if u < accept_prob {
        // Accept: keep current state (already there in SoA buffers)
        accept_flags[idx] = 1u;

        // ENTROPY EXTRACTION (at high temperature)
        if uniforms.temperature > 1.0 {
            var entropy = current_scalars.entropy_bits;
            for (var d = 0u; d < dim; d = d + 1u) {
                let pos_val = f32(positions[d * pc + idx]);
                let pos_bits = bitcast<u32>(pos_val * 1e6);
                entropy = entropy ^ hash(pos_bits + idx * 1000u + d);
            }
            var s = current_scalars;
            s.entropy_bits = entropy;
            scalars[idx] = s;
            entropy_output[idx] = entropy;
        }
    } else {
        // Reject: restore initial position from proposal (but negate momentum)
        for (var d = 0u; d < dim; d = d + 1u) {
            positions[d * pc + idx] = proposal_pos[d * pc + idx];
            velocities[d * pc + idx] = -proposal_vel[d * pc + idx];  // Negate momentum
        }
        scalars[idx] = initial_scalars;
        accept_flags[idx] = 0u;
    }
}
