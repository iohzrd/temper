// Unified Thermodynamic Particle System
//
// This shader demonstrates that entropy generation, Bayesian sampling, and optimization
// are all points on a temperature continuum:
//
// T >> 1   : Entropy mode    - Particles explore chaotically, extract randomness
// T ~ 0.1  : Sampling mode   - SVGD/Langevin samples from posterior distribution
// T → 0    : Optimize mode   - Particles converge to loss minima
//
// The key insight: These are NOT three different algorithms, but ONE algorithm
// with a temperature parameter that controls the exploration/exploitation tradeoff.

// Enable f16 support for reduced memory bandwidth (~50% memory reduction)
enable f16;

// MAX_DIMENSIONS = 64 to support deeper neural networks
// Struct size: 64*2 + 4 + 4 = 136 bytes (down from 272 with f32 positions)
// ~50% memory reduction = better cache utilization and bandwidth
struct Particle {
    pos: array<f16, 64>,     // Position in parameter space (NN weights) - f16 for bandwidth
    energy: f32,             // Current loss/energy value (f32 for precision)
    entropy_bits: u32,       // Accumulated entropy bits (for extraction)
}

// Helper to read position as f32 for computation
fn get_pos(p: Particle, d: u32) -> f32 {
    return f32(p.pos[d]);
}

// Helper to get position array as f32 for loss functions
fn get_pos_array(p: Particle, dim: u32) -> array<f32, 64> {
    var result: array<f32, 64>;
    for (var i = 0u; i < dim; i = i + 1u) {
        result[i] = f32(p.pos[i]);
    }
    return result;
}

struct Uniforms {
    particle_count: u32,
    dim: u32,
    gamma: f32,              // Friction coefficient
    temperature: f32,        // THE KEY PARAMETER: controls mode
    repulsion_strength: f32,
    kernel_bandwidth: f32,
    dt: f32,
    seed: u32,
    // Mode indicators (computed from temperature)
    mode: u32,               // 0=optimize, 1=sample, 2=entropy
    // Loss function selector
    // 0=neural_net_2d, 1=multimodal, 2=rosenbrock, 3=rastrigin, 4=ackley, 5=sphere
    loss_fn: u32,
    // Repulsion samples: 0 = skip repulsion, >0 = sample this many particles (O(nK) instead of O(n²))
    repulsion_samples: u32,
    _pad1: f32,
}

// Training data (same neural net task)
const TRAIN_X: array<f32, 10> = array<f32, 10>(
    -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5
);
const TRAIN_Y: array<f32, 10> = array<f32, 10>(
    -1.93, -1.79, -1.52, -0.93, 0.0, 0.93, 1.52, 1.79, 1.93, 1.97
);
const TRAIN_SIZE: u32 = 10u;

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read_write> repulsion: array<Particle>;
@group(0) @binding(3) var<storage, read_write> entropy_output: array<u32>;

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

// Stub functions for custom expressions - these are overridden when using with_expr()
// They must exist for the shader to compile even when LossFunction::Custom isn't used
fn custom_loss(pos: array<f32, 64>, dim: u32) -> f32 {
    // Default: sphere function
    var sum = 0.0;
    for (var i = 0u; i < dim; i = i + 1u) {
        sum = sum + pos[i] * pos[i];
    }
    return sum;
}

fn custom_gradient(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {
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
fn multimodal_loss_nd(pos: array<f32, 64>, dim: u32) -> f32 {
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
fn multimodal_gradient_nd(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {
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

fn mlp_xor_forward(pos: array<f32, 64>, input: vec2<f32>) -> f32 {
    // Layer 1: input (2) -> hidden (2) with tanh
    let h0 = tanh(pos[0] * input.x + pos[1] * input.y + pos[4]);
    let h1 = tanh(pos[2] * input.x + pos[3] * input.y + pos[5]);

    // Layer 2: hidden (2) -> output (1) with sigmoid
    let out = sigmoid(pos[6] * h0 + pos[7] * h1 + pos[8]);
    return out;
}

fn mlp_xor_loss(pos: array<f32, 64>) -> f32 {
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

fn mlp_xor_gradient(pos: array<f32, 64>, d_idx: u32) -> f32 {
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
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

fn mlp_spiral_loss(pos: array<f32, 64>) -> f32 {
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

fn mlp_spiral_gradient(pos: array<f32, 64>, d_idx: u32) -> f32 {
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

fn mlp_deep_forward(pos: array<f32, 64>, input: vec2<f32>) -> f32 {
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

fn mlp_deep_loss(pos: array<f32, 64>) -> f32 {
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

fn mlp_deep_gradient(pos: array<f32, 64>, d_idx: u32) -> f32 {
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
// CLASSIC OPTIMIZATION BENCHMARK FUNCTIONS
// ============================================================================

// Rosenbrock function (N-dimensional)
// Global minimum: f(1,1,...,1) = 0
// Famous "banana valley" - tests ability to follow narrow curved valleys
fn rosenbrock_loss(pos: array<f32, 64>, dim: u32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < dim - 1u; i = i + 1u) {
        let x_i = pos[i];
        let x_i1 = pos[i + 1u];
        sum = sum + 100.0 * (x_i1 - x_i * x_i) * (x_i1 - x_i * x_i) + (1.0 - x_i) * (1.0 - x_i);
    }
    return sum;
}

fn rosenbrock_gradient(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {
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
const PI: f32 = 3.14159265359;

fn rastrigin_loss(pos: array<f32, 64>, dim: u32) -> f32 {
    var sum = 10.0 * f32(dim);
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        sum = sum + x * x - 10.0 * cos(2.0 * PI * x);
    }
    return sum;
}

fn rastrigin_gradient(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {
    let x = pos[d_idx];
    return 2.0 * x + 20.0 * PI * sin(2.0 * PI * x);
}

// Ackley function (N-dimensional)
// Global minimum: f(0,0,...,0) = 0
// Nearly flat outer region with deep hole at center - tests exploitation
fn ackley_loss(pos: array<f32, 64>, dim: u32) -> f32 {
    let a = 20.0;
    let b = 0.2;
    let c = 2.0 * PI;

    var sum_sq = 0.0;
    var sum_cos = 0.0;
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        sum_sq = sum_sq + x * x;
        sum_cos = sum_cos + cos(c * x);
    }

    let n = f32(dim);
    return -a * exp(-b * sqrt(sum_sq / n)) - exp(sum_cos / n) + a + 2.71828182845;
}

fn ackley_gradient(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {
    let a = 20.0;
    let b = 0.2;
    let c = 2.0 * PI;

    var sum_sq = 0.0;
    var sum_cos = 0.0;
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        sum_sq = sum_sq + x * x;
        sum_cos = sum_cos + cos(c * x);
    }

    let n = f32(dim);
    let x_d = pos[d_idx];
    let sqrt_term = sqrt(sum_sq / n);

    // Gradient of first term: a * b * x_d / (n * sqrt_term) * exp(-b * sqrt_term)
    var grad = 0.0;
    if sqrt_term > 0.0001 {
        grad = a * b * x_d / (n * sqrt_term) * exp(-b * sqrt_term);
    }

    // Gradient of second term: c * sin(c * x_d) / n * exp(sum_cos / n)
    grad = grad + c * sin(c * x_d) / n * exp(sum_cos / n);

    return grad;
}

// Sphere function (N-dimensional) - simple convex baseline
// Global minimum: f(0,0,...,0) = 0
fn sphere_loss(pos: array<f32, 64>, dim: u32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < dim; i = i + 1u) {
        sum = sum + pos[i] * pos[i];
    }
    return sum;
}

fn sphere_gradient(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {
    return 2.0 * pos[d_idx];
}

// Schwefel function (N-dimensional) - deceptive global minimum far from origin
// Global minimum: f(420.9687, ..., 420.9687) = 0
// The global minimum is far from the origin, and local minima are deceptive
fn schwefel_loss(pos: array<f32, 64>, dim: u32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < dim; i = i + 1u) {
        let x = pos[i];
        sum = sum + x * sin(sqrt(abs(x)));
    }
    return 418.9829 * f32(dim) - sum;
}

fn schwefel_gradient(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {
    let x = pos[d_idx];
    let abs_x = abs(x);

    if abs_x < 0.0001 {
        return 0.0;
    }

    let sqrt_abs_x = sqrt(abs_x);
    let sign_x = select(-1.0, 1.0, x >= 0.0);

    // d/dx [x * sin(sqrt(|x|))] = sin(sqrt(|x|)) + x * cos(sqrt(|x|)) * (sign(x) / (2 * sqrt(|x|)))
    // = sin(sqrt(|x|)) + sign(x) * sqrt(|x|) * cos(sqrt(|x|)) / 2
    let grad = sin(sqrt_abs_x) + sign_x * sqrt_abs_x * cos(sqrt_abs_x) / 2.0;

    // Negative because schwefel_loss = 418.9829*n - sum
    return -grad;
}

// ============================================================================

// Pass 1: Compute pairwise repulsion (SVGD kernel gradient)
// Optimized: Uses subsampling when repulsion_samples > 0 to achieve O(nK) instead of O(n²)
@compute @workgroup_size(64)
fn compute_repulsion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.particle_count {
        return;
    }

    // Skip repulsion entirely if repulsion_samples is 0 (pure optimization mode)
    if uniforms.repulsion_samples == 0u {
        for (var d = 0u; d < 64u; d = d + 1u) {
            repulsion[idx].pos[d] = f16(0.0);
        }
        return;
    }

    var p_i = particles[idx];
    let h_sq = uniforms.kernel_bandwidth * uniforms.kernel_bandwidth;

    // Accumulate repulsion in f32 for precision
    var rep: array<f32, 64>;
    for (var d = 0u; d < 64u; d = d + 1u) {
        rep[d] = 0.0;
    }

    // Determine how many particles to sample
    let sample_count = min(uniforms.repulsion_samples, uniforms.particle_count - 1u);

    // Use subsampling: randomly select K particles instead of all N
    // This reduces O(n²) to O(nK) complexity
    for (var s = 0u; s < sample_count; s = s + 1u) {
        // Hash-based random particle selection
        // Different particles get different random sequences via idx mixing
        let rng_seed = uniforms.seed + idx * 1337u + s * 7919u;
        let j = hash(rng_seed) % uniforms.particle_count;

        // Skip self
        if j == idx {
            continue;
        }

        var p_j = particles[j];

        var dist_sq = 0.0;
        for (var d = 0u; d < uniforms.dim; d = d + 1u) {
            let diff = f32(p_i.pos[d]) - f32(p_j.pos[d]);
            dist_sq = dist_sq + diff * diff;
        }
        let k = exp(-dist_sq / (2.0 * h_sq));

        // Kernel gradient pushes particles apart
        for (var d = 0u; d < uniforms.dim; d = d + 1u) {
            let diff = f32(p_i.pos[d]) - f32(p_j.pos[d]);
            rep[d] = rep[d] - k * diff / h_sq;
        }
    }

    // Scale by effective sample count (importance sampling correction)
    let effective_n = f32(sample_count);
    for (var d = 0u; d < uniforms.dim; d = d + 1u) {
        repulsion[idx].pos[d] = f16(rep[d] * uniforms.repulsion_strength / effective_n);
    }
}

// Pass 2: Update particles with temperature-controlled dynamics
@compute @workgroup_size(64)
fn update_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.particle_count {
        return;
    }

    var p = particles[idx];

    // Convert f16 positions to f32 for computation
    var pos_f32 = get_pos_array(p, uniforms.dim);

    // Compute loss based on selected loss function
    // 0=neural_net_2d, 1=multimodal, 2=rosenbrock, 3=rastrigin, 4=ackley, 5=sphere
    var loss_grad_2d = vec2<f32>(0.0, 0.0);

    switch uniforms.loss_fn {
        case 0u: {
            // Neural net 2D (original)
            p.energy = nn_loss_2d(pos_f32[0], pos_f32[1]);
            loss_grad_2d = nn_gradient_2d(pos_f32[0], pos_f32[1]);
        }
        case 1u: {
            // Multimodal N-D
            p.energy = multimodal_loss_nd(pos_f32, uniforms.dim);
        }
        case 2u: {
            // Rosenbrock
            p.energy = rosenbrock_loss(pos_f32, uniforms.dim);
        }
        case 3u: {
            // Rastrigin
            p.energy = rastrigin_loss(pos_f32, uniforms.dim);
        }
        case 4u: {
            // Ackley
            p.energy = ackley_loss(pos_f32, uniforms.dim);
        }
        case 5u: {
            // Sphere
            p.energy = sphere_loss(pos_f32, uniforms.dim);
        }
        case 6u: {
            // MLP XOR (real neural network)
            p.energy = mlp_xor_loss(pos_f32);
        }
        case 7u: {
            // MLP Spiral classification
            p.energy = mlp_spiral_loss(pos_f32);
        }
        case 8u: {
            // Deep MLP (3-layer) on circles
            p.energy = mlp_deep_loss(pos_f32);
        }
        case 9u: {
            // Schwefel - deceptive with far-off global minimum
            p.energy = schwefel_loss(pos_f32, uniforms.dim);
        }
        case 10u: {
            // Custom expression-based loss function (injected at shader compile time)
            p.energy = custom_loss(pos_f32, uniforms.dim);
        }
        default: {
            p.energy = nn_loss_2d(pos_f32[0], pos_f32[1]);
            loss_grad_2d = nn_gradient_2d(pos_f32[0], pos_f32[1]);
        }
    }

    // THE UNIFIED UPDATE EQUATION:
    // dx = -γ∇E(x)·dt + repulsion·dt + √(2γT·dt)·dW
    //
    // When T >> 1: Noise dominates → chaotic exploration (entropy mode)
    // When T ~ 0.1: Balance → samples from exp(-E/T) (sampling mode)
    // When T → 0: Gradient dominates → optimization (optimize mode)

    let noise_scale = sqrt(2.0 * uniforms.gamma * uniforms.temperature * uniforms.dt);

    // Repulsion scaling: stronger at medium T, weaker at extremes
    // This keeps particles diverse during sampling, less important for pure optimization
    let repulsion_scale = select(
        1.0,
        select(0.1, 1.0, uniforms.temperature > 0.001),
        uniforms.temperature < 10.0
    );

    for (var d = 0u; d < uniforms.dim; d = d + 1u) {
        // Generate noise
        let seed1 = uniforms.seed + idx * 17u + d * 31u;
        let seed2 = uniforms.seed + idx * 37u + d * 53u + 12345u;
        let noise = randn(seed1, seed2);

        // Get gradient for this dimension based on loss function
        var grad = 0.0;
        switch uniforms.loss_fn {
            case 0u: {
                // Neural net 2D
                if d == 0u {
                    grad = loss_grad_2d.x;
                } else if d == 1u {
                    grad = loss_grad_2d.y;
                }
            }
            case 1u: {
                // Multimodal N-D
                grad = multimodal_gradient_nd(pos_f32, uniforms.dim, d);
            }
            case 2u: {
                // Rosenbrock
                grad = rosenbrock_gradient(pos_f32, uniforms.dim, d);
            }
            case 3u: {
                // Rastrigin
                grad = rastrigin_gradient(pos_f32, uniforms.dim, d);
            }
            case 4u: {
                // Ackley
                grad = ackley_gradient(pos_f32, uniforms.dim, d);
            }
            case 5u: {
                // Sphere
                grad = sphere_gradient(pos_f32, uniforms.dim, d);
            }
            case 6u: {
                // MLP XOR
                grad = mlp_xor_gradient(pos_f32, d);
            }
            case 7u: {
                // MLP Spiral
                grad = mlp_spiral_gradient(pos_f32, d);
            }
            case 8u: {
                // Deep MLP
                grad = mlp_deep_gradient(pos_f32, d);
            }
            case 9u: {
                // Schwefel
                grad = schwefel_gradient(pos_f32, uniforms.dim, d);
            }
            case 10u: {
                // Custom expression-based gradient
                grad = custom_gradient(pos_f32, uniforms.dim, d);
            }
            default: {
                if d == 0u {
                    grad = loss_grad_2d.x;
                } else if d == 1u {
                    grad = loss_grad_2d.y;
                }
            }
        }

        // Clip large gradients to prevent explosion (esp. for Rosenbrock)
        let grad_clipped = clamp(grad, -10.0, 10.0);

        // The unified update (all arithmetic in f32)
        let grad_term = -uniforms.gamma * grad_clipped;
        let repulsion_term = f32(repulsion[idx].pos[d]) * repulsion_scale;
        let noise_term = noise_scale * noise;

        // Update position in f32
        pos_f32[d] = pos_f32[d] + (grad_term + repulsion_term) * uniforms.dt + noise_term;

        // Clamp based on loss function (Schwefel needs larger domain)
        if uniforms.loss_fn == 9u {
            // Schwefel: global minimum at ~420.97
            pos_f32[d] = clamp(pos_f32[d], -500.0, 500.0);
        } else {
            pos_f32[d] = clamp(pos_f32[d], -5.0, 5.0);
        }

        // Write back as f16
        p.pos[d] = f16(pos_f32[d]);
    }

    // ENTROPY EXTRACTION (at high temperature)
    // When T is high, particle positions are chaotic - extract bits!
    if uniforms.temperature > 1.0 {
        // Mix position bits into entropy accumulator (use f32 positions for bit extraction)
        var entropy = p.entropy_bits;
        for (var d = 0u; d < uniforms.dim; d = d + 1u) {
            let pos_bits = bitcast<u32>(pos_f32[d] * 1e6);
            entropy = entropy ^ hash(pos_bits + idx * 1000u + d);
        }
        p.entropy_bits = entropy;

        // Write to entropy output buffer
        entropy_output[idx] = entropy;
    }

    particles[idx] = p;
}

// Pass 3: Collect statistics (optional, for visualization)
@compute @workgroup_size(64)
fn collect_stats(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Could compute mean, variance, entropy estimates here
    // For now, stats are computed on CPU after readback
}
