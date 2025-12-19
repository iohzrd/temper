//! Standard optimization test landscapes
//!
//! This module provides common 2D test functions for optimization visualization.
//! Each landscape includes:
//! - Rust energy evaluation (for rendering)
//! - WGSL loss and gradient functions (for GPU computation)

use crate::expr::{Expr, custom_wgsl};

/// Bounds for 2D landscape visualization
pub const X_MIN: f32 = -5.0;
pub const X_MAX: f32 = 5.0;
pub const Y_MIN: f32 = -5.0;
pub const Y_MAX: f32 = 5.0;

/// Standard 2D optimization test landscapes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Landscape {
    /// Rastrigin function: many local minima, global minimum at origin
    Rastrigin,
    /// Double well: two minima at (-2, 0) and (2, 0)
    DoubleWell,
    /// Four well: four minima at corners (±2, ±2)
    FourWell,
    /// Rosenbrock: banana-shaped valley, minimum at (1, 1)
    Rosenbrock,
    /// Himmelblau: four equal minima
    Himmelblau,
}

impl Landscape {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Landscape::Rastrigin => "Rastrigin",
            Landscape::DoubleWell => "Double Well",
            Landscape::FourWell => "Four Well",
            Landscape::Rosenbrock => "Rosenbrock",
            Landscape::Himmelblau => "Himmelblau",
        }
    }

    /// Cycle to the next landscape
    pub fn next(&self) -> Self {
        match self {
            Landscape::Rastrigin => Landscape::DoubleWell,
            Landscape::DoubleWell => Landscape::FourWell,
            Landscape::FourWell => Landscape::Rosenbrock,
            Landscape::Rosenbrock => Landscape::Himmelblau,
            Landscape::Himmelblau => Landscape::Rastrigin,
        }
    }

    /// Get all landscapes
    pub fn all() -> &'static [Landscape] {
        &[
            Landscape::Rastrigin,
            Landscape::DoubleWell,
            Landscape::FourWell,
            Landscape::Rosenbrock,
            Landscape::Himmelblau,
        ]
    }

    /// Get WGSL expression for GPU computation
    pub fn expr(&self) -> Expr {
        match self {
            Landscape::Rastrigin => rastrigin_expr(),
            Landscape::DoubleWell => double_well_expr(),
            Landscape::FourWell => four_well_expr(),
            Landscape::Rosenbrock => rosenbrock_expr(),
            Landscape::Himmelblau => himmelblau_expr(),
        }
    }

    /// Evaluate energy at (x, y) for rendering
    pub fn energy(&self, x: f32, y: f32) -> f32 {
        match self {
            Landscape::Rastrigin => {
                let a = 10.0;
                let pi = std::f32::consts::PI;
                a * 2.0 + (x * x - a * (2.0 * pi * x).cos()) + (y * y - a * (2.0 * pi * y).cos())
            }
            Landscape::DoubleWell => {
                let r1 = ((x + 2.0).powi(2) + y.powi(2)).sqrt();
                let r2 = ((x - 2.0).powi(2) + y.powi(2)).sqrt();
                r1.min(r2).powi(2)
            }
            Landscape::FourWell => {
                let r1 = ((x + 2.0).powi(2) + (y + 2.0).powi(2)).sqrt();
                let r2 = ((x + 2.0).powi(2) + (y - 2.0).powi(2)).sqrt();
                let r3 = ((x - 2.0).powi(2) + (y + 2.0).powi(2)).sqrt();
                let r4 = ((x - 2.0).powi(2) + (y - 2.0).powi(2)).sqrt();
                r1.min(r2).min(r3).min(r4).powi(2)
            }
            Landscape::Rosenbrock => {
                let a = 1.0;
                let b = 10.0; // Scaled down for numerical stability
                (a - x).powi(2) + b * (y - x.powi(2)).powi(2)
            }
            Landscape::Himmelblau => (x.powi(2) + y - 11.0).powi(2) + (x + y.powi(2) - 7.0).powi(2),
        }
    }

    /// Get description of the landscape
    pub fn description(&self) -> &'static str {
        match self {
            Landscape::Rastrigin => "Many local minima, global at origin",
            Landscape::DoubleWell => "Two minima at (-2,0) and (2,0)",
            Landscape::FourWell => "Four minima at (±2, ±2)",
            Landscape::Rosenbrock => "Banana-shaped valley, minimum at (1,1)",
            Landscape::Himmelblau => "Four equal minima",
        }
    }
}

// =============================================================================
// WGSL Loss Function Generators
// =============================================================================

fn rastrigin_expr() -> Expr {
    let loss_code = r#"
fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    let a = 10.0;
    let pi = 3.14159265359;
    var sum = a * 2.0;
    for (var i = 0u; i < 2u; i = i + 1u) {
        let x = pos[i];
        sum = sum + x * x - a * cos(2.0 * pi * x);
    }
    let bound = 5.0;
    var penalty = 0.0;
    if abs(pos[0]) > bound { penalty = penalty + 50.0 * (abs(pos[0]) - bound) * (abs(pos[0]) - bound); }
    if abs(pos[1]) > bound { penalty = penalty + 50.0 * (abs(pos[1]) - bound) * (abs(pos[1]) - bound); }
    return sum + penalty;
}
"#;

    let grad_code = r#"
fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    if d_idx >= 2u { return 0.0; }
    let a = 10.0;
    let pi = 3.14159265359;
    let x = pos[d_idx];
    let bound = 5.0;
    var grad = 2.0 * x + a * 2.0 * pi * sin(2.0 * pi * x);
    if x > bound { grad = grad + 100.0 * (x - bound); }
    if x < -bound { grad = grad - 100.0 * (-bound - x); }
    return clamp(grad, -10.0, 10.0);
}
"#;

    custom_wgsl(loss_code, grad_code)
}

fn double_well_expr() -> Expr {
    let loss_code = r#"
fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    let x = pos[0];
    let y = pos[1];
    let r1 = sqrt((x + 2.0) * (x + 2.0) + y * y);
    let r2 = sqrt((x - 2.0) * (x - 2.0) + y * y);
    let r = min(r1, r2);
    let bound = 5.0;
    var penalty = 0.0;
    if abs(x) > bound { penalty = penalty + 50.0 * (abs(x) - bound) * (abs(x) - bound); }
    if abs(y) > bound { penalty = penalty + 50.0 * (abs(y) - bound) * (abs(y) - bound); }
    return r * r + penalty;
}
"#;

    let grad_code = r#"
fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    if d_idx >= 2u { return 0.0; }
    let x = pos[0];
    let y = pos[1];
    let bound = 5.0;
    let r1 = sqrt((x + 2.0) * (x + 2.0) + y * y);
    let r2 = sqrt((x - 2.0) * (x - 2.0) + y * y);
    var grad = 0.0;
    if r1 < r2 {
        if d_idx == 0u { grad = 2.0 * (x + 2.0); }
        else { grad = 2.0 * y; }
    } else {
        if d_idx == 0u { grad = 2.0 * (x - 2.0); }
        else { grad = 2.0 * y; }
    }
    let p = pos[d_idx];
    if p > bound { grad = grad + 100.0 * (p - bound); }
    if p < -bound { grad = grad - 100.0 * (-bound - p); }
    return clamp(grad, -10.0, 10.0);
}
"#;

    custom_wgsl(loss_code, grad_code)
}

fn four_well_expr() -> Expr {
    let loss_code = r#"
fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    let x = pos[0];
    let y = pos[1];
    let r1 = sqrt((x + 2.0) * (x + 2.0) + (y + 2.0) * (y + 2.0));
    let r2 = sqrt((x + 2.0) * (x + 2.0) + (y - 2.0) * (y - 2.0));
    let r3 = sqrt((x - 2.0) * (x - 2.0) + (y + 2.0) * (y + 2.0));
    let r4 = sqrt((x - 2.0) * (x - 2.0) + (y - 2.0) * (y - 2.0));
    let r = min(min(r1, r2), min(r3, r4));
    let bound = 5.0;
    var penalty = 0.0;
    if abs(x) > bound { penalty = penalty + 50.0 * (abs(x) - bound) * (abs(x) - bound); }
    if abs(y) > bound { penalty = penalty + 50.0 * (abs(y) - bound) * (abs(y) - bound); }
    return r * r + penalty;
}
"#;

    let grad_code = r#"
fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    if d_idx >= 2u { return 0.0; }
    let x = pos[0];
    let y = pos[1];
    let bound = 5.0;
    let r1 = sqrt((x + 2.0) * (x + 2.0) + (y + 2.0) * (y + 2.0));
    let r2 = sqrt((x + 2.0) * (x + 2.0) + (y - 2.0) * (y - 2.0));
    let r3 = sqrt((x - 2.0) * (x - 2.0) + (y + 2.0) * (y + 2.0));
    let r4 = sqrt((x - 2.0) * (x - 2.0) + (y - 2.0) * (y - 2.0));
    var cx = -2.0; var cy = -2.0; var min_r = r1;
    if r2 < min_r { cx = -2.0; cy = 2.0; min_r = r2; }
    if r3 < min_r { cx = 2.0; cy = -2.0; min_r = r3; }
    if r4 < min_r { cx = 2.0; cy = 2.0; }
    var grad = 0.0;
    if d_idx == 0u { grad = 2.0 * (x - cx); }
    else { grad = 2.0 * (y - cy); }
    let p = pos[d_idx];
    if p > bound { grad = grad + 100.0 * (p - bound); }
    if p < -bound { grad = grad - 100.0 * (-bound - p); }
    return clamp(grad, -10.0, 10.0);
}
"#;

    custom_wgsl(loss_code, grad_code)
}

fn rosenbrock_expr() -> Expr {
    let loss_code = r#"
fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    let x = pos[0];
    let y = pos[1];
    let a = 1.0;
    let b = 10.0;  // Scaled down from 100 for stability
    let rosenbrock = (a - x) * (a - x) + b * (y - x * x) * (y - x * x);
    let bound = 5.0;
    var penalty = 0.0;
    if abs(x) > bound { penalty = penalty + 20.0 * (abs(x) - bound) * (abs(x) - bound); }
    if abs(y) > bound { penalty = penalty + 20.0 * (abs(y) - bound) * (abs(y) - bound); }
    return rosenbrock + penalty;
}
"#;

    let grad_code = r#"
fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    if d_idx >= 2u { return 0.0; }
    let x = pos[0];
    let y = pos[1];
    let a = 1.0;
    let b = 10.0;
    let bound = 5.0;
    var grad = 0.0;
    if d_idx == 0u {
        grad = -2.0 * (a - x) - 4.0 * b * x * (y - x * x);
        if x > bound { grad = grad + 40.0 * (x - bound); }
        if x < -bound { grad = grad - 40.0 * (-bound - x); }
    } else {
        grad = 2.0 * b * (y - x * x);
        if y > bound { grad = grad + 40.0 * (y - bound); }
        if y < -bound { grad = grad - 40.0 * (-bound - y); }
    }
    return clamp(grad, -10.0, 10.0);
}
"#;

    custom_wgsl(loss_code, grad_code)
}

fn himmelblau_expr() -> Expr {
    let loss_code = r#"
fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    let x = pos[0];
    let y = pos[1];
    let a = x * x + y - 11.0;
    let b = x + y * y - 7.0;
    let bound = 5.0;
    var penalty = 0.0;
    if abs(x) > bound { penalty = penalty + 50.0 * (abs(x) - bound) * (abs(x) - bound); }
    if abs(y) > bound { penalty = penalty + 50.0 * (abs(y) - bound) * (abs(y) - bound); }
    return a * a + b * b + penalty;
}
"#;

    let grad_code = r#"
fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    if d_idx >= 2u { return 0.0; }
    let x = pos[0];
    let y = pos[1];
    let bound = 5.0;
    let a = x * x + y - 11.0;
    let b = x + y * y - 7.0;
    var grad = 0.0;
    if d_idx == 0u {
        grad = 4.0 * x * a + 2.0 * b;
    } else {
        grad = 2.0 * a + 4.0 * y * b;
    }
    let p = pos[d_idx];
    if p > bound { grad = grad + 100.0 * (p - bound); }
    if p < -bound { grad = grad - 100.0 * (-bound - p); }
    return clamp(grad, -10.0, 10.0);
}
"#;

    custom_wgsl(loss_code, grad_code)
}
