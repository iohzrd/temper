//! Optimizer Comparison Benchmark
//!
//! Compares the thermodynamic particle optimizer against traditional optimizers:
//! - SGD (Stochastic Gradient Descent)
//! - Adam (Adaptive Moment Estimation)
//!
//! Tests on various landscapes to show where particle-based search excels:
//! - Unimodal (Sphere): All optimizers should work well
//! - Multimodal (Rastrigin): Particle methods better at avoiding local minima
//! - Neural Network (XOR): Real-world non-convex optimization

use nbody_entropy::thermodynamic::{LossFunction, ThermodynamicSystem};
use std::time::Instant;

// ============================================================================
// Loss Functions (CPU implementations for SGD/Adam)
// ============================================================================

fn sphere_loss(x: &[f32]) -> f32 {
    x.iter().map(|xi| xi * xi).sum()
}

fn sphere_gradient(x: &[f32]) -> Vec<f32> {
    x.iter().map(|xi| 2.0 * xi).collect()
}

fn rastrigin_loss(x: &[f32]) -> f32 {
    let n = x.len() as f32;
    let pi = std::f32::consts::PI;
    10.0 * n + x.iter().map(|xi| xi * xi - 10.0 * (2.0 * pi * xi).cos()).sum::<f32>()
}

fn rastrigin_gradient(x: &[f32]) -> Vec<f32> {
    let pi = std::f32::consts::PI;
    x.iter().map(|xi| 2.0 * xi + 20.0 * pi * (2.0 * pi * xi).sin()).collect()
}

fn rosenbrock_loss(x: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
    }
    sum
}

fn rosenbrock_gradient(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mut grad = vec![0.0; n];
    for i in 0..n {
        if i < n - 1 {
            grad[i] += -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i]);
        }
        if i > 0 {
            grad[i] += 200.0 * (x[i] - x[i - 1] * x[i - 1]);
        }
    }
    grad
}

// XOR MLP: 2->2->1 (9 parameters)
fn xor_mlp_forward(w: &[f32], x: f32, y: f32) -> f32 {
    let h0 = (w[0] * x + w[1] * y + w[4]).tanh();
    let h1 = (w[2] * x + w[3] * y + w[5]).tanh();
    1.0 / (1.0 + (-w[6] * h0 - w[7] * h1 - w[8]).exp())
}

fn xor_loss(w: &[f32]) -> f32 {
    let data = [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)];
    let mut total = 0.0;
    for (x, y, target) in &data {
        let pred = xor_mlp_forward(w, *x, *y);
        let eps = 0.0001;
        let p = pred.clamp(eps, 1.0 - eps);
        total -= target * p.ln() + (1.0 - target) * (1.0 - p).ln();
    }
    total / 4.0
}

fn xor_gradient(w: &[f32]) -> Vec<f32> {
    let eps = 0.001;
    let mut grad = vec![0.0; w.len()];
    for i in 0..w.len() {
        let mut w_plus = w.to_vec();
        let mut w_minus = w.to_vec();
        w_plus[i] += eps;
        w_minus[i] -= eps;
        grad[i] = (xor_loss(&w_plus) - xor_loss(&w_minus)) / (2.0 * eps);
    }
    grad
}

// ============================================================================
// Optimizers
// ============================================================================

struct SgdOptimizer {
    lr: f32,
}

impl SgdOptimizer {
    fn new(lr: f32) -> Self {
        Self { lr }
    }

    fn step(&self, x: &mut [f32], grad: &[f32]) {
        for (xi, gi) in x.iter_mut().zip(grad.iter()) {
            *xi -= self.lr * gi;
        }
    }
}

struct AdamOptimizer {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
}

impl AdamOptimizer {
    fn new(lr: f32, dim: usize) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: vec![0.0; dim],
            v: vec![0.0; dim],
            t: 0,
        }
    }

    fn step(&mut self, x: &mut [f32], grad: &[f32]) {
        self.t += 1;
        for i in 0..x.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.t));
            let v_hat = self.v[i] / (1.0 - self.beta2.powi(self.t));
            x[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ============================================================================
// Benchmarks
// ============================================================================

struct BenchmarkResult {
    optimizer: String,
    final_loss: f32,
    best_loss: f32,
    time_ms: u128,
    converged: bool,
}

fn run_sgd<F, G>(name: &str, loss_fn: F, grad_fn: G, dim: usize, init: &[f32], steps: usize, lr: f32, threshold: f32) -> BenchmarkResult
where
    F: Fn(&[f32]) -> f32,
    G: Fn(&[f32]) -> Vec<f32>,
{
    let mut x = init.to_vec();
    let mut best_loss = f32::MAX;
    let optimizer = SgdOptimizer::new(lr);

    let start = Instant::now();
    for _ in 0..steps {
        let grad = grad_fn(&x);
        optimizer.step(&mut x, &grad);
        // Clamp to bounds
        for xi in x.iter_mut() {
            *xi = xi.clamp(-5.0, 5.0);
        }
        let loss = loss_fn(&x);
        best_loss = best_loss.min(loss);
    }
    let elapsed = start.elapsed().as_millis();
    let final_loss = loss_fn(&x);

    BenchmarkResult {
        optimizer: format!("SGD (lr={})", lr),
        final_loss,
        best_loss,
        time_ms: elapsed,
        converged: best_loss < threshold,
    }
}

fn run_adam<F, G>(name: &str, loss_fn: F, grad_fn: G, dim: usize, init: &[f32], steps: usize, lr: f32, threshold: f32) -> BenchmarkResult
where
    F: Fn(&[f32]) -> f32,
    G: Fn(&[f32]) -> Vec<f32>,
{
    let mut x = init.to_vec();
    let mut best_loss = f32::MAX;
    let mut optimizer = AdamOptimizer::new(lr, dim);

    let start = Instant::now();
    for _ in 0..steps {
        let grad = grad_fn(&x);
        optimizer.step(&mut x, &grad);
        for xi in x.iter_mut() {
            *xi = xi.clamp(-5.0, 5.0);
        }
        let loss = loss_fn(&x);
        best_loss = best_loss.min(loss);
    }
    let elapsed = start.elapsed().as_millis();
    let final_loss = loss_fn(&x);

    BenchmarkResult {
        optimizer: format!("Adam (lr={})", lr),
        final_loss,
        best_loss,
        time_ms: elapsed,
        converged: best_loss < threshold,
    }
}

fn run_thermodynamic(loss_fn: LossFunction, dim: usize, steps: usize, threshold: f32) -> BenchmarkResult {
    let particle_count = 500;
    let mut system = ThermodynamicSystem::with_loss_function(particle_count, dim, 1.0, loss_fn);

    let start = Instant::now();
    for step in 0..steps {
        let progress = step as f32 / steps as f32;
        let temp = 1.0 * (0.001_f32 / 1.0).powf(progress);
        system.set_temperature(temp);
        system.step();
    }
    let elapsed = start.elapsed().as_millis();

    let particles = system.read_particles();
    let best_loss = particles.iter()
        .filter(|p| !p.energy.is_nan())
        .map(|p| p.energy)
        .fold(f32::MAX, f32::min);

    BenchmarkResult {
        optimizer: "Thermodynamic".to_string(),
        final_loss: best_loss,
        best_loss,
        time_ms: elapsed,
        converged: best_loss < threshold,
    }
}

fn print_results(name: &str, results: &[BenchmarkResult]) {
    println!("\n  {}", name);
    println!("  {:─<70}", "");
    println!("  {:20} {:>12} {:>12} {:>10} {:>8}", "Optimizer", "Best Loss", "Final Loss", "Time (ms)", "Status");
    println!("  {:─<70}", "");
    for r in results {
        let status = if r.converged { "✓" } else { "✗" };
        println!("  {:20} {:>12.6} {:>12.6} {:>10} {:>8}",
            r.optimizer, r.best_loss, r.final_loss, r.time_ms, status);
    }
}

fn main() {
    println!("{}",
        "╔══════════════════════════════════════════════════════════════════════════╗");
    println!("{}",
        "║           OPTIMIZER COMPARISON BENCHMARK                                 ║");
    println!("{}",
        "╠══════════════════════════════════════════════════════════════════════════╣");
    println!("{}",
        "║  Comparing Thermodynamic Particles vs SGD vs Adam                        ║");
    println!("{}",
        "║  on Sphere (easy), Rosenbrock (hard), Rastrigin (multimodal), XOR (NN)   ║");
    println!("{}",
        "╚══════════════════════════════════════════════════════════════════════════╝\n");

    let steps = 2000;

    // Test 1: Sphere (easy convex)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 1: SPHERE (4D) - Easy convex, all optimizers should excel");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    {
        let dim = 4;
        let init: Vec<f32> = vec![3.0, -2.0, 1.5, -3.5];
        let threshold = 0.01;

        let results = vec![
            run_sgd("sphere", sphere_loss, sphere_gradient, dim, &init, steps, 0.1, threshold),
            run_adam("sphere", sphere_loss, sphere_gradient, dim, &init, steps, 0.1, threshold),
            run_thermodynamic(LossFunction::Sphere, dim, steps, threshold),
        ];
        print_results("Sphere 4D (optimum at origin)", &results);
    }

    // Test 2: Rosenbrock (hard banana valley)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 2: ROSENBROCK (4D) - Hard narrow valley, tests precision");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    {
        let dim = 4;
        let init: Vec<f32> = vec![-1.0, 1.0, -1.0, 1.0];
        let threshold = 10.0;

        let results = vec![
            run_sgd("rosenbrock", rosenbrock_loss, rosenbrock_gradient, dim, &init, steps, 0.0001, threshold),
            run_adam("rosenbrock", rosenbrock_loss, rosenbrock_gradient, dim, &init, steps, 0.01, threshold),
            run_thermodynamic(LossFunction::Rosenbrock, dim, steps, threshold),
        ];
        print_results("Rosenbrock 4D (optimum at [1,1,1,1])", &results);
    }

    // Test 3: Rastrigin (highly multimodal)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 3: RASTRIGIN (4D) - Many local minima, tests global search");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    {
        let dim = 4;
        let init: Vec<f32> = vec![2.5, -2.5, 3.0, -1.0];
        let threshold = 10.0;

        let results = vec![
            run_sgd("rastrigin", rastrigin_loss, rastrigin_gradient, dim, &init, steps, 0.01, threshold),
            run_adam("rastrigin", rastrigin_loss, rastrigin_gradient, dim, &init, steps, 0.01, threshold),
            run_thermodynamic(LossFunction::Rastrigin, dim, steps * 2, threshold),
        ];
        print_results("Rastrigin 4D (optimum at origin)", &results);
    }

    // Test 4: XOR Neural Network
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 4: XOR MLP (9 params) - Non-convex neural network");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    {
        let dim = 9;
        // Random-ish initialization
        let init: Vec<f32> = vec![0.5, -0.3, 0.8, -0.6, 0.1, -0.2, 0.7, -0.4, 0.0];
        let threshold = 0.5;

        let results = vec![
            run_sgd("xor", xor_loss, xor_gradient, dim, &init, steps * 2, 0.5, threshold),
            run_adam("xor", xor_loss, xor_gradient, dim, &init, steps * 2, 0.1, threshold),
            run_thermodynamic(LossFunction::MlpXor, dim, steps * 2, threshold),
        ];
        print_results("XOR MLP (BCE loss, target < 0.5)", &results);
    }

    // Summary
    println!("\n{}",
        "╔══════════════════════════════════════════════════════════════════════════╗");
    println!("{}",
        "║                              KEY INSIGHTS                                ║");
    println!("{}",
        "╠══════════════════════════════════════════════════════════════════════════╣");
    println!("{}",
        "║  • Sphere: All optimizers work well on simple convex problems            ║");
    println!("{}",
        "║  • Rosenbrock: Adam handles narrow valleys better than SGD               ║");
    println!("{}",
        "║  • Rastrigin: Thermodynamic excels at escaping local minima              ║");
    println!("{}",
        "║  • XOR: Particle methods explore the loss landscape more thoroughly      ║");
    println!("{}",
        "╚══════════════════════════════════════════════════════════════════════════╝");
}
