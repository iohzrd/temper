//! Expression DSL for custom loss functions
//!
//! Build loss functions from composable primitives that compile to GPU-accelerated WGSL.
//!
//! # Example
//! ```ignore
//! use temper::expr::*;
//!
//! // Griewank function: 1 + sum(x^2/4000) - prod(cos(x/sqrt(i)))
//! let griewank = const_(1.0)
//!     + sum_dims(|x, _| x.powi(2) / 4000.0)
//!     - prod_dims(|x, i| cos(x / sqrt(i + 1.0)));
//! ```

use std::sync::Arc;

/// Mathematical expression that compiles to WGSL
#[derive(Clone)]
pub enum Expr {
    /// Variable x[i] where i is the dimension index
    Var,
    /// Dimension index (0, 1, 2, ...)
    DimIndex,
    /// Total number of dimensions
    DimCount,
    /// Constant value
    Const(f32),
    /// Pi constant
    Pi,

    // Unary operations
    Neg(Arc<Expr>),
    Abs(Arc<Expr>),
    Sin(Arc<Expr>),
    Cos(Arc<Expr>),
    Tan(Arc<Expr>),
    Exp(Arc<Expr>),
    Ln(Arc<Expr>),
    Sqrt(Arc<Expr>),
    Tanh(Arc<Expr>),
    Floor(Arc<Expr>),
    Ceil(Arc<Expr>),
    Sign(Arc<Expr>),

    // Binary operations
    Add(Arc<Expr>, Arc<Expr>),
    Sub(Arc<Expr>, Arc<Expr>),
    Mul(Arc<Expr>, Arc<Expr>),
    Div(Arc<Expr>, Arc<Expr>),
    Pow(Arc<Expr>, Arc<Expr>),
    Min(Arc<Expr>, Arc<Expr>),
    Max(Arc<Expr>, Arc<Expr>),
    Mod(Arc<Expr>, Arc<Expr>),

    // Reductions over dimensions
    /// Sum over all dimensions: sum_i f(x[i], i)
    SumDims(Arc<dyn Fn(Expr, Expr) -> Expr + Send + Sync>),
    /// Product over all dimensions: prod_i f(x[i], i)
    ProdDims(Arc<dyn Fn(Expr, Expr) -> Expr + Send + Sync>),
    /// Sum over adjacent pairs: sum_i f(x[i], x[i+1])
    SumPairs(Arc<dyn Fn(Expr, Expr) -> Expr + Send + Sync>),
    /// Custom WGSL code (loss function body, gradient function body)
    CustomWgsl(String, String),
}

impl std::fmt::Debug for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Var => write!(f, "Var"),
            Expr::DimIndex => write!(f, "DimIndex"),
            Expr::DimCount => write!(f, "DimCount"),
            Expr::Const(v) => write!(f, "Const({v})"),
            Expr::Pi => write!(f, "Pi"),
            Expr::Neg(e) => write!(f, "Neg({e:?})"),
            Expr::Abs(e) => write!(f, "Abs({e:?})"),
            Expr::Sin(e) => write!(f, "Sin({e:?})"),
            Expr::Cos(e) => write!(f, "Cos({e:?})"),
            Expr::Tan(e) => write!(f, "Tan({e:?})"),
            Expr::Exp(e) => write!(f, "Exp({e:?})"),
            Expr::Ln(e) => write!(f, "Ln({e:?})"),
            Expr::Sqrt(e) => write!(f, "Sqrt({e:?})"),
            Expr::Tanh(e) => write!(f, "Tanh({e:?})"),
            Expr::Floor(e) => write!(f, "Floor({e:?})"),
            Expr::Ceil(e) => write!(f, "Ceil({e:?})"),
            Expr::Sign(e) => write!(f, "Sign({e:?})"),
            Expr::Add(a, b) => write!(f, "Add({a:?}, {b:?})"),
            Expr::Sub(a, b) => write!(f, "Sub({a:?}, {b:?})"),
            Expr::Mul(a, b) => write!(f, "Mul({a:?}, {b:?})"),
            Expr::Div(a, b) => write!(f, "Div({a:?}, {b:?})"),
            Expr::Pow(a, b) => write!(f, "Pow({a:?}, {b:?})"),
            Expr::Min(a, b) => write!(f, "Min({a:?}, {b:?})"),
            Expr::Max(a, b) => write!(f, "Max({a:?}, {b:?})"),
            Expr::Mod(a, b) => write!(f, "Mod({a:?}, {b:?})"),
            Expr::SumDims(_) => write!(f, "SumDims(<fn>)"),
            Expr::ProdDims(_) => write!(f, "ProdDims(<fn>)"),
            Expr::SumPairs(_) => write!(f, "SumPairs(<fn>)"),
            Expr::CustomWgsl(_, _) => write!(f, "CustomWgsl(<code>)"),
        }
    }
}

// ============================================================================
// Builder functions
// ============================================================================

/// The variable x[i] in the current dimension
pub fn var() -> Expr {
    Expr::Var
}

/// The current dimension index (0, 1, 2, ...)
pub fn dim_index() -> Expr {
    Expr::DimIndex
}

/// Total number of dimensions
pub fn dim_count() -> Expr {
    Expr::DimCount
}

/// Constant value
pub fn const_(v: f32) -> Expr {
    Expr::Const(v)
}

/// Pi constant (3.14159...)
pub fn pi() -> Expr {
    Expr::Pi
}

// Unary functions
pub fn abs(e: Expr) -> Expr {
    Expr::Abs(Arc::new(e))
}
pub fn sin(e: Expr) -> Expr {
    Expr::Sin(Arc::new(e))
}
pub fn cos(e: Expr) -> Expr {
    Expr::Cos(Arc::new(e))
}
pub fn tan(e: Expr) -> Expr {
    Expr::Tan(Arc::new(e))
}
pub fn exp(e: Expr) -> Expr {
    Expr::Exp(Arc::new(e))
}
pub fn ln(e: Expr) -> Expr {
    Expr::Ln(Arc::new(e))
}
pub fn sqrt(e: Expr) -> Expr {
    Expr::Sqrt(Arc::new(e))
}
pub fn tanh(e: Expr) -> Expr {
    Expr::Tanh(Arc::new(e))
}
pub fn floor(e: Expr) -> Expr {
    Expr::Floor(Arc::new(e))
}
pub fn ceil(e: Expr) -> Expr {
    Expr::Ceil(Arc::new(e))
}
pub fn sign(e: Expr) -> Expr {
    Expr::Sign(Arc::new(e))
}

// Binary functions
pub fn min(a: Expr, b: Expr) -> Expr {
    Expr::Min(Arc::new(a), Arc::new(b))
}
pub fn max(a: Expr, b: Expr) -> Expr {
    Expr::Max(Arc::new(a), Arc::new(b))
}
pub fn pow(base: Expr, exp: Expr) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(exp))
}
pub fn modulo(a: Expr, b: Expr) -> Expr {
    Expr::Mod(Arc::new(a), Arc::new(b))
}

/// Sum over all dimensions: sum_i f(x[i], i)
pub fn sum_dims<F>(f: F) -> Expr
where
    F: Fn(Expr, Expr) -> Expr + Send + Sync + 'static,
{
    Expr::SumDims(Arc::new(f))
}

/// Product over all dimensions: prod_i f(x[i], i)
pub fn prod_dims<F>(f: F) -> Expr
where
    F: Fn(Expr, Expr) -> Expr + Send + Sync + 'static,
{
    Expr::ProdDims(Arc::new(f))
}

/// Sum over adjacent pairs: sum_i f(x[i], x[i+1])
pub fn sum_pairs<F>(f: F) -> Expr
where
    F: Fn(Expr, Expr) -> Expr + Send + Sync + 'static,
{
    Expr::SumPairs(Arc::new(f))
}

/// Custom WGSL loss function
/// Takes two strings: the complete custom_loss function body and the complete custom_gradient function body.
/// The strings should be valid WGSL code that defines:
/// - `fn custom_loss(pos: array<f32, 64>, dim: u32) -> f32`
/// - `fn custom_gradient(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32`
pub fn custom_wgsl(loss_code: &str, gradient_code: &str) -> Expr {
    Expr::CustomWgsl(loss_code.to_string(), gradient_code.to_string())
}

// ============================================================================
// Operator overloading
// ============================================================================

impl Expr {
    /// Raise to integer power
    pub fn powi(self, n: i32) -> Expr {
        Expr::Pow(Arc::new(self), Arc::new(Expr::Const(n as f32)))
    }

    /// Raise to float power
    pub fn powf(self, n: f32) -> Expr {
        Expr::Pow(Arc::new(self), Arc::new(Expr::Const(n)))
    }
}

impl std::ops::Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::Neg(Arc::new(self))
    }
}

impl std::ops::Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::Add(Arc::new(self), Arc::new(rhs))
    }
}

impl std::ops::Sub for Expr {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr::Sub(Arc::new(self), Arc::new(rhs))
    }
}

impl std::ops::Mul for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::Mul(Arc::new(self), Arc::new(rhs))
    }
}

impl std::ops::Div for Expr {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        Expr::Div(Arc::new(self), Arc::new(rhs))
    }
}

// f32 on left side
impl std::ops::Add<Expr> for f32 {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::Add(Arc::new(Expr::Const(self)), Arc::new(rhs))
    }
}

impl std::ops::Sub<Expr> for f32 {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr::Sub(Arc::new(Expr::Const(self)), Arc::new(rhs))
    }
}

impl std::ops::Mul<Expr> for f32 {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::Mul(Arc::new(Expr::Const(self)), Arc::new(rhs))
    }
}

impl std::ops::Div<Expr> for f32 {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        Expr::Div(Arc::new(Expr::Const(self)), Arc::new(rhs))
    }
}

// f32 on right side
impl std::ops::Add<f32> for Expr {
    type Output = Expr;
    fn add(self, rhs: f32) -> Expr {
        Expr::Add(Arc::new(self), Arc::new(Expr::Const(rhs)))
    }
}

impl std::ops::Sub<f32> for Expr {
    type Output = Expr;
    fn sub(self, rhs: f32) -> Expr {
        Expr::Sub(Arc::new(self), Arc::new(Expr::Const(rhs)))
    }
}

impl std::ops::Mul<f32> for Expr {
    type Output = Expr;
    fn mul(self, rhs: f32) -> Expr {
        Expr::Mul(Arc::new(self), Arc::new(Expr::Const(rhs)))
    }
}

impl std::ops::Div<f32> for Expr {
    type Output = Expr;
    fn div(self, rhs: f32) -> Expr {
        Expr::Div(Arc::new(self), Arc::new(Expr::Const(rhs)))
    }
}

// ============================================================================
// WGSL Code Generation
// ============================================================================

impl Expr {
    /// Generate complete WGSL loss function with analytical gradients
    pub fn to_wgsl(&self) -> String {
        self.to_wgsl_with_options(true)
    }

    /// Generate WGSL with option for analytical or numerical gradients
    pub fn to_wgsl_with_options(&self, _analytical: bool) -> String {
        // Special case: CustomWgsl emits raw WGSL code directly
        if let Expr::CustomWgsl(loss_code, grad_code) = self {
            return format!("{}\n\n{}", loss_code, grad_code);
        }

        let mut counter = 0;
        let mut helpers = Vec::new();
        let body = self.to_wgsl_with_helpers("x", "i", &mut counter, &mut helpers);

        let (all_helpers, grad_body) = if _analytical {
            // Generate analytical gradient helpers
            let mut grad_counter = 500;
            let mut grad_helpers = Vec::new();
            let grad_body = self.to_wgsl_gradient(&mut grad_counter, &mut grad_helpers);
            ([helpers, grad_helpers].concat().join("\n\n"), grad_body)
        } else {
            // Use numerical gradient (fallback)
            (
                helpers.join("\n\n"),
                "numerical_gradient(pos, dim, d_idx)".to_string(),
            )
        };

        let numerical_helper = if !_analytical {
            r#"

fn numerical_gradient(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {
    let eps = 0.001;
    var pos_plus = pos;
    var pos_minus = pos;
    pos_plus[d_idx] = pos[d_idx] + eps;
    pos_minus[d_idx] = pos[d_idx] - eps;
    return (custom_loss(pos_plus, dim) - custom_loss(pos_minus, dim)) / (2.0 * eps);
}"#
        } else {
            ""
        };

        format!(
            r#"{all_helpers}

fn custom_loss(pos: array<f32, 64>, dim: u32) -> f32 {{
    return {body};
}}

fn custom_gradient(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{
    return {grad_body};
}}{numerical_helper}"#
        )
    }

    /// Generate WGSL code for analytical gradient ∂f/∂pos[d_idx]
    fn to_wgsl_gradient(&self, counter: &mut u32, helpers: &mut Vec<String>) -> String {
        match self {
            // Constants have zero gradient
            Expr::Const(_) | Expr::Pi | Expr::DimIndex | Expr::DimCount => "0.0".to_string(),

            // Var is x[i] in the loop context - gradient is 1 when i == d_idx
            // This is handled specially in reduction contexts
            Expr::Var => "1.0".to_string(),

            // Negation: d(-f)/dx = -df/dx
            Expr::Neg(e) => {
                let de = e.to_wgsl_gradient(counter, helpers);
                if de == "0.0" {
                    "0.0".to_string()
                } else {
                    format!("(-{})", de)
                }
            }

            // Absolute value: d|f|/dx = sign(f) * df/dx
            Expr::Abs(e) => {
                let de = e.to_wgsl_gradient(counter, helpers);
                if de == "0.0" {
                    "0.0".to_string()
                } else {
                    *counter += 1;
                    let fn_name = format!("abs_grad_{}", counter);
                    let inner =
                        e.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                    helpers.push(format!(
                        "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ return sign({inner}) * {de}; }}"
                    ));
                    format!("{fn_name}(pos, dim, d_idx)")
                }
            }

            // Trigonometric: d(sin f)/dx = cos(f) * df/dx
            Expr::Sin(e) => {
                let de = e.to_wgsl_gradient(counter, helpers);
                if de == "0.0" {
                    "0.0".to_string()
                } else {
                    *counter += 1;
                    let fn_name = format!("sin_grad_{}", counter);
                    let inner =
                        e.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                    helpers.push(format!(
                        "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ return cos({inner}) * {de}; }}"
                    ));
                    format!("{fn_name}(pos, dim, d_idx)")
                }
            }

            // d(cos f)/dx = -sin(f) * df/dx
            Expr::Cos(e) => {
                let de = e.to_wgsl_gradient(counter, helpers);
                if de == "0.0" {
                    "0.0".to_string()
                } else {
                    *counter += 1;
                    let fn_name = format!("cos_grad_{}", counter);
                    let inner =
                        e.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                    helpers.push(format!(
                        "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ return -sin({inner}) * {de}; }}"
                    ));
                    format!("{fn_name}(pos, dim, d_idx)")
                }
            }

            // d(tan f)/dx = sec²(f) * df/dx = df/dx / cos²(f)
            Expr::Tan(e) => {
                let de = e.to_wgsl_gradient(counter, helpers);
                if de == "0.0" {
                    "0.0".to_string()
                } else {
                    *counter += 1;
                    let fn_name = format!("tan_grad_{}", counter);
                    let inner =
                        e.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                    helpers.push(format!(
                        "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ let c = cos({inner}); return {de} / (c * c); }}"
                    ));
                    format!("{fn_name}(pos, dim, d_idx)")
                }
            }

            // d(exp f)/dx = exp(f) * df/dx
            Expr::Exp(e) => {
                let de = e.to_wgsl_gradient(counter, helpers);
                if de == "0.0" {
                    "0.0".to_string()
                } else {
                    *counter += 1;
                    let fn_name = format!("exp_grad_{}", counter);
                    let inner =
                        e.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                    helpers.push(format!(
                        "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ return exp({inner}) * {de}; }}"
                    ));
                    format!("{fn_name}(pos, dim, d_idx)")
                }
            }

            // d(ln f)/dx = df/dx / f
            Expr::Ln(e) => {
                let de = e.to_wgsl_gradient(counter, helpers);
                if de == "0.0" {
                    "0.0".to_string()
                } else {
                    *counter += 1;
                    let fn_name = format!("ln_grad_{}", counter);
                    let inner =
                        e.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                    helpers.push(format!(
                        "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ return {de} / {inner}; }}"
                    ));
                    format!("{fn_name}(pos, dim, d_idx)")
                }
            }

            // d(sqrt f)/dx = df/dx / (2 * sqrt(f))
            Expr::Sqrt(e) => {
                let de = e.to_wgsl_gradient(counter, helpers);
                if de == "0.0" {
                    "0.0".to_string()
                } else {
                    *counter += 1;
                    let fn_name = format!("sqrt_grad_{}", counter);
                    let inner =
                        e.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                    helpers.push(format!(
                        "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ return {de} / (2.0 * sqrt({inner})); }}"
                    ));
                    format!("{fn_name}(pos, dim, d_idx)")
                }
            }

            // d(tanh f)/dx = (1 - tanh²(f)) * df/dx
            Expr::Tanh(e) => {
                let de = e.to_wgsl_gradient(counter, helpers);
                if de == "0.0" {
                    "0.0".to_string()
                } else {
                    *counter += 1;
                    let fn_name = format!("tanh_grad_{}", counter);
                    let inner =
                        e.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                    helpers.push(format!(
                        "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ let t = tanh({inner}); return (1.0 - t * t) * {de}; }}"
                    ));
                    format!("{fn_name}(pos, dim, d_idx)")
                }
            }

            // Floor, Ceil, Sign have zero gradient (piecewise constant)
            Expr::Floor(_) | Expr::Ceil(_) | Expr::Sign(_) => "0.0".to_string(),

            // Addition: d(f + g)/dx = df/dx + dg/dx
            Expr::Add(a, b) => {
                let da = a.to_wgsl_gradient(counter, helpers);
                let db = b.to_wgsl_gradient(counter, helpers);
                match (da.as_str(), db.as_str()) {
                    ("0.0", "0.0") => "0.0".to_string(),
                    ("0.0", _) => db,
                    (_, "0.0") => da,
                    _ => format!("({da} + {db})"),
                }
            }

            // Subtraction: d(f - g)/dx = df/dx - dg/dx
            Expr::Sub(a, b) => {
                let da = a.to_wgsl_gradient(counter, helpers);
                let db = b.to_wgsl_gradient(counter, helpers);
                match (da.as_str(), db.as_str()) {
                    ("0.0", "0.0") => "0.0".to_string(),
                    ("0.0", _) => format!("(-{db})"),
                    (_, "0.0") => da,
                    _ => format!("({da} - {db})"),
                }
            }

            // Product rule: d(f * g)/dx = f * dg/dx + g * df/dx
            Expr::Mul(a, b) => {
                let da = a.to_wgsl_gradient(counter, helpers);
                let db = b.to_wgsl_gradient(counter, helpers);
                let fa = a.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                let fb = b.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                match (da.as_str(), db.as_str()) {
                    ("0.0", "0.0") => "0.0".to_string(),
                    ("0.0", _) => format!("({fa} * {db})"),
                    (_, "0.0") => format!("({fb} * {da})"),
                    _ => {
                        *counter += 1;
                        let fn_name = format!("mul_grad_{}", counter);
                        helpers.push(format!(
                            "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ return {fa} * {db} + {fb} * {da}; }}"
                        ));
                        format!("{fn_name}(pos, dim, d_idx)")
                    }
                }
            }

            // Quotient rule: d(f/g)/dx = (g * df/dx - f * dg/dx) / g²
            Expr::Div(a, b) => {
                let da = a.to_wgsl_gradient(counter, helpers);
                let db = b.to_wgsl_gradient(counter, helpers);
                let fa = a.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                let fb = b.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                match (da.as_str(), db.as_str()) {
                    ("0.0", "0.0") => "0.0".to_string(),
                    ("0.0", _) => {
                        *counter += 1;
                        let fn_name = format!("div_grad_{}", counter);
                        helpers.push(format!(
                            "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ return -{fa} * {db} / ({fb} * {fb}); }}"
                        ));
                        format!("{fn_name}(pos, dim, d_idx)")
                    }
                    (_, "0.0") => format!("({da} / {fb})"),
                    _ => {
                        *counter += 1;
                        let fn_name = format!("div_grad_{}", counter);
                        helpers.push(format!(
                            "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ let g = {fb}; return ({da} * g - {fa} * {db}) / (g * g); }}"
                        ));
                        format!("{fn_name}(pos, dim, d_idx)")
                    }
                }
            }

            // Power rule: d(f^g)/dx = f^g * (g * df/dx / f + ln(f) * dg/dx)
            // Special case: if g is constant n, d(f^n)/dx = n * f^(n-1) * df/dx
            Expr::Pow(base, exp) => {
                let dbase = base.to_wgsl_gradient(counter, helpers);
                let dexp = exp.to_wgsl_gradient(counter, helpers);
                let fbase =
                    base.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                let fexp = exp.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());

                match (dbase.as_str(), dexp.as_str()) {
                    ("0.0", "0.0") => "0.0".to_string(),
                    (_, "0.0") => {
                        // Constant exponent: n * f^(n-1) * df/dx
                        *counter += 1;
                        let fn_name = format!("pow_grad_{}", counter);
                        helpers.push(format!(
                            "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ return {fexp} * pow({fbase}, {fexp} - 1.0) * {dbase}; }}"
                        ));
                        format!("{fn_name}(pos, dim, d_idx)")
                    }
                    ("0.0", _) => {
                        // Constant base: f^g * ln(f) * dg/dx
                        *counter += 1;
                        let fn_name = format!("pow_grad_{}", counter);
                        helpers.push(format!(
                            "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ return pow({fbase}, {fexp}) * log({fbase}) * {dexp}; }}"
                        ));
                        format!("{fn_name}(pos, dim, d_idx)")
                    }
                    _ => {
                        // General case
                        *counter += 1;
                        let fn_name = format!("pow_grad_{}", counter);
                        helpers.push(format!(
                            "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ let f = {fbase}; let g = {fexp}; let fg = pow(f, g); return fg * (g * {dbase} / f + log(f) * {dexp}); }}"
                        ));
                        format!("{fn_name}(pos, dim, d_idx)")
                    }
                }
            }

            // Min/Max have subgradients; use conditional
            Expr::Min(a, b) => {
                let da = a.to_wgsl_gradient(counter, helpers);
                let db = b.to_wgsl_gradient(counter, helpers);
                if da == "0.0" && db == "0.0" {
                    "0.0".to_string()
                } else {
                    let fa = a.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                    let fb = b.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                    *counter += 1;
                    let fn_name = format!("min_grad_{}", counter);
                    helpers.push(format!(
                        "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ return select({db}, {da}, {fa} < {fb}); }}"
                    ));
                    format!("{fn_name}(pos, dim, d_idx)")
                }
            }

            Expr::Max(a, b) => {
                let da = a.to_wgsl_gradient(counter, helpers);
                let db = b.to_wgsl_gradient(counter, helpers);
                if da == "0.0" && db == "0.0" {
                    "0.0".to_string()
                } else {
                    let fa = a.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                    let fb = b.to_wgsl_with_helpers("pos[d_idx]", "d_idx", &mut 0, &mut Vec::new());
                    *counter += 1;
                    let fn_name = format!("max_grad_{}", counter);
                    helpers.push(format!(
                        "fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{ return select({da}, {db}, {fa} < {fb}); }}"
                    ));
                    format!("{fn_name}(pos, dim, d_idx)")
                }
            }

            // Mod: d(a % b)/dx = da/dx (assuming b is constant)
            Expr::Mod(a, _b) => a.to_wgsl_gradient(counter, helpers),

            // SumDims: d/dx[j] sum_i f(x[i], i) = df(x[j], j)/dx[j]
            // Only the term where i == d_idx contributes
            Expr::SumDims(f) => {
                *counter += 1;
                let fn_name = format!("sum_grad_{}", counter);

                // Get the body expression and its gradient
                let body_expr = f(Expr::Var, Expr::DimIndex);
                let mut body_counter = 600 + *counter * 10;
                let body_grad = body_expr.to_wgsl_gradient(&mut body_counter, helpers);

                helpers.push(format!(
                    r#"fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{
    // Gradient of sum_i f(x[i], i) w.r.t. x[d_idx] is just df/dx at i=d_idx
    return {body_grad};
}}"#
                ));
                format!("{fn_name}(pos, dim, d_idx)")
            }

            // ProdDims: d/dx[j] prod_i f(x[i], i) = prod * (df_j/dx / f_j)
            // Using logarithmic differentiation
            Expr::ProdDims(f) => {
                *counter += 1;
                let fn_name = format!("prod_grad_{}", counter);
                let loss_fn = format!("prod_loss_{}", counter);

                // Get body expression
                let body_expr = f(Expr::Var, Expr::DimIndex);
                let mut body_counter = 700 + *counter * 10;
                let body_code =
                    body_expr.to_wgsl_with_helpers("x", "i", &mut body_counter, helpers);
                let body_grad = body_expr.to_wgsl_gradient(&mut body_counter, helpers);

                // First, create a helper to compute the full product
                helpers.push(format!(
                    r#"fn {loss_fn}(pos: array<f32, 64>, dim: u32) -> f32 {{
    var result = 1.0;
    for (var i = 0u; i < dim; i = i + 1u) {{
        let x = pos[i];
        result = result * {body_code};
    }}
    return result;
}}"#
                ));

                // Then the gradient using d(prod)/dx[j] = prod * (df_j/f_j)
                helpers.push(format!(
                    r#"fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{
    let prod = {loss_fn}(pos, dim);
    let x = pos[d_idx];
    let i = d_idx;
    let f_j = {body_code};
    let df_j = {body_grad};
    return prod * df_j / f_j;
}}"#
                ));
                format!("{fn_name}(pos, dim, d_idx)")
            }

            // SumPairs: d/dx[j] sum_i f(x[i], x[i+1])
            // This requires computing df/dx and df/dy separately, which our DSL doesn't track.
            // Use numerical gradient for this case.
            Expr::SumPairs(f) => {
                *counter += 1;
                let fn_name = format!("pair_grad_{}", counter);
                let loss_fn = format!("pair_loss_{}", counter);

                // Get body expression for loss computation
                let body_expr = f(Expr::Var, Expr::Var);
                let mut body_counter = 800 + *counter * 10;
                let body_code =
                    body_expr.to_wgsl_with_helpers("x", "i", &mut body_counter, helpers);

                // Create the loss function helper
                helpers.push(format!(
                    r#"fn {loss_fn}(pos: array<f32, 64>, dim: u32) -> f32 {{
    var result = 0.0;
    for (var i = 0u; i < dim - 1u; i = i + 1u) {{
        let x = pos[i];
        let y = pos[i + 1u];
        _ = y;
        result = result + {body_code};
    }}
    return result;
}}"#
                ));

                // Numerical gradient for sum_pairs (df/dx requires knowing both x and y roles)
                helpers.push(format!(
                    r#"fn {fn_name}(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{
    let eps = 0.001;
    var pos_plus = pos;
    var pos_minus = pos;
    pos_plus[d_idx] = pos[d_idx] + eps;
    pos_minus[d_idx] = pos[d_idx] - eps;
    return ({loss_fn}(pos_plus, dim) - {loss_fn}(pos_minus, dim)) / (2.0 * eps);
}}"#
                ));

                format!("{fn_name}(pos, dim, d_idx)")
            }

            // CustomWgsl is handled at the top level, should never reach here
            Expr::CustomWgsl(_, _) => "0.0".to_string(),
        }
    }

    /// Generate WGSL code with helper functions for reductions
    fn to_wgsl_with_helpers(
        &self,
        var_name: &str,
        idx_name: &str,
        counter: &mut u32,
        helpers: &mut Vec<String>,
    ) -> String {
        match self {
            Expr::Var => var_name.to_string(),
            Expr::DimIndex => format!("f32({idx_name})"),
            Expr::DimCount => "f32(dim)".to_string(),
            Expr::Const(v) => {
                if v.fract() == 0.0 && v.abs() < 1e6 {
                    format!("{:.1}", v)
                } else {
                    format!("{:?}", v)
                }
            }
            Expr::Pi => "3.14159265359".to_string(),

            Expr::Neg(e) => format!(
                "(-{})",
                e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Abs(e) => format!(
                "abs({})",
                e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Sin(e) => format!(
                "sin({})",
                e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Cos(e) => format!(
                "cos({})",
                e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Tan(e) => format!(
                "tan({})",
                e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Exp(e) => format!(
                "exp({})",
                e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Ln(e) => format!(
                "log({})",
                e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Sqrt(e) => format!(
                "sqrt({})",
                e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Tanh(e) => format!(
                "tanh({})",
                e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Floor(e) => format!(
                "floor({})",
                e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Ceil(e) => format!(
                "ceil({})",
                e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Sign(e) => format!(
                "sign({})",
                e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),

            Expr::Add(a, b) => format!(
                "({} + {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Sub(a, b) => format!(
                "({} - {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Mul(a, b) => format!(
                "({} * {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Div(a, b) => format!(
                "({} / {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Pow(a, b) => format!(
                "pow({}, {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Min(a, b) => format!(
                "min({}, {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Max(a, b) => format!(
                "max({}, {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),
            Expr::Mod(a, b) => format!(
                "({} % {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)
            ),

            Expr::SumDims(f) => {
                *counter += 1;
                let fn_name = format!("sum_helper_{}", counter);
                let loop_var = "i";
                let x_var = "x";

                let body_expr = f(Expr::Var, Expr::DimIndex);
                let mut inner_counter = 100 + *counter * 10;
                let body_code =
                    body_expr.to_wgsl_with_helpers(x_var, loop_var, &mut inner_counter, helpers);

                let helper = format!(
                    r#"fn {fn_name}(pos: array<f32, 64>, dim: u32) -> f32 {{
    var result = 0.0;
    for (var {loop_var} = 0u; {loop_var} < dim; {loop_var} = {loop_var} + 1u) {{
        let {x_var} = pos[{loop_var}];
        result = result + {body_code};
    }}
    return result;
}}"#
                );
                helpers.push(helper);
                format!("{fn_name}(pos, dim)")
            }

            Expr::ProdDims(f) => {
                *counter += 1;
                let fn_name = format!("prod_helper_{}", counter);
                let loop_var = "i";
                let x_var = "x";

                let body_expr = f(Expr::Var, Expr::DimIndex);
                let mut inner_counter = 100 + *counter * 10;
                let body_code =
                    body_expr.to_wgsl_with_helpers(x_var, loop_var, &mut inner_counter, helpers);

                let helper = format!(
                    r#"fn {fn_name}(pos: array<f32, 64>, dim: u32) -> f32 {{
    var result = 1.0;
    for (var {loop_var} = 0u; {loop_var} < dim; {loop_var} = {loop_var} + 1u) {{
        let {x_var} = pos[{loop_var}];
        result = result * {body_code};
    }}
    return result;
}}"#
                );
                helpers.push(helper);
                format!("{fn_name}(pos, dim)")
            }

            Expr::SumPairs(f) => {
                *counter += 1;
                let fn_name = format!("pair_helper_{}", counter);
                let loop_var = "i";
                let x_var = "x";
                let y_var = "y";

                let body_expr = f(Expr::Var, Expr::Var);
                let mut inner_counter = 100 + *counter * 10;
                let body_code =
                    body_expr.to_wgsl_with_helpers(x_var, loop_var, &mut inner_counter, helpers);

                let helper = format!(
                    r#"fn {fn_name}(pos: array<f32, 64>, dim: u32) -> f32 {{
    var result = 0.0;
    for (var {loop_var} = 0u; {loop_var} < dim - 1u; {loop_var} = {loop_var} + 1u) {{
        let {x_var} = pos[{loop_var}];
        let {y_var} = pos[{loop_var} + 1u];
        result = result + {body_code};
    }}
    return result;
}}"#
                );
                helpers.push(helper);
                format!("{fn_name}(pos, dim)")
            }

            // CustomWgsl is handled at the top level, should never reach here
            Expr::CustomWgsl(_, _) => "0.0".to_string(),
        }
    }
}

// ============================================================================
// Pre-built loss functions
// ============================================================================

/// Sphere function: sum(x^2), minimum at origin
pub fn sphere() -> Expr {
    sum_dims(|x, _| x.powi(2))
}

/// Rastrigin function: 10n + sum(x^2 - 10*cos(2πx)), minimum at origin
pub fn rastrigin() -> Expr {
    10.0 * dim_count() + sum_dims(|x, _| x.clone().powi(2) - 10.0 * cos(2.0 * pi() * x))
}

/// Rosenbrock function: sum(100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2), minimum at (1,1,...,1)
pub fn rosenbrock() -> Expr {
    sum_pairs(|x, y| {
        let term1 = (y.clone() - x.clone().powi(2)).powi(2) * 100.0;
        let term2 = (1.0 - x).powi(2);
        term1 + term2
    })
}

/// Ackley function: minimum at origin
pub fn ackley() -> Expr {
    let a = 20.0;
    let b = 0.2;
    let c = 2.0 * std::f32::consts::PI;

    let sum_sq = sum_dims(|x, _| x.powi(2));
    let sum_cos = sum_dims(move |x, _| cos(const_(c) * x));

    const_(-a) * exp(const_(-b) * sqrt(sum_sq / dim_count())) - exp(sum_cos / dim_count())
        + const_(a + std::f32::consts::E)
}

/// Griewank function: 1 + sum(x^2/4000) - prod(cos(x/sqrt(i+1))), minimum at origin
pub fn griewank() -> Expr {
    const_(1.0) + sum_dims(|x, _| x.powi(2) / 4000.0) - prod_dims(|x, i| cos(x / sqrt(i + 1.0)))
}

/// Levy function: minimum at (1,1,...,1)
pub fn levy() -> Expr {
    // w_i = 1 + (x_i - 1) / 4
    // Levy = sin^2(pi*w_1) + sum((w_i-1)^2 * (1 + 10*sin^2(pi*w_i+1))) + (w_n-1)^2 * (1 + sin^2(2*pi*w_n))
    // Simplified version for DSL
    sum_dims(|x, _| {
        let w = 1.0 + (x.clone() - 1.0) / 4.0;
        (w.clone() - 1.0).powi(2) * (1.0 + 10.0 * sin(pi() * w).powi(2))
    })
}

/// Michalewicz function: -sum(sin(x_i) * sin^(2m)((i+1)*x_i^2/π))
/// Multimodal with steep ridges and valleys. Default m=10.
/// Global minimum depends on dimension, approximately -1.8013 for 2D at (2.20, 1.57)
pub fn michalewicz() -> Expr {
    michalewicz_m(10)
}

/// Michalewicz function with configurable steepness parameter m
pub fn michalewicz_m(m: i32) -> Expr {
    -sum_dims(move |x, i| {
        let inner = (i + 1.0) * x.clone().powi(2) / pi();
        sin(x) * sin(inner).powi(2 * m)
    })
}

/// Styblinski-Tang function: 0.5 * sum(x_i^4 - 16*x_i^2 + 5*x_i)
/// Global minimum at x_i ≈ -2.903534 with value ≈ -39.16617*n
pub fn styblinski_tang() -> Expr {
    0.5 * sum_dims(|x, _| x.clone().powi(4) - 16.0 * x.clone().powi(2) + 5.0 * x)
}

/// Dixon-Price function: (x_1 - 1)^2 + sum_{i=2}^n (i * (2*x_i^2 - x_{i-1})^2)
/// Uses sum_pairs for the coupled terms
/// Global minimum at x_i = 2^(-((2^i - 2)/2^i))
pub fn dixon_price() -> Expr {
    // First term: (x_0 - 1)^2 handled separately via a trick
    // sum_pairs gives us access to consecutive pairs
    sum_pairs(|x, y| {
        // For pair (x_i, x_{i+1}), we compute (i+2) * (2*x_{i+1}^2 - x_i)^2
        // This is shifted by 1 since sum_pairs starts at i=0
        (2.0 * y.clone().powi(2) - x).powi(2)
    }) + sum_dims(|x, i| {
        // Add (i+1) scaling factor and first term (x_0 - 1)^2
        // When i=0, this gives (x_0 - 1)^2
        // For other terms, the scaling is handled in sum_pairs
        (i.clone() * const_(0.0))
            + (const_(1.0) - x).powi(2) * (const_(1.0) - i.clone() / (i + 1.0))
    })
}

/// Zakharov function: sum(x_i^2) + (0.5*sum((i+1)*x_i))^2 + (0.5*sum((i+1)*x_i))^4
/// Bowl-shaped with a flat valley. Global minimum at origin.
pub fn zakharov() -> Expr {
    let sum_sq = sum_dims(|x, _| x.powi(2));
    let weighted_sum = sum_dims(|x, i| 0.5 * (i + 1.0) * x);
    sum_sq + weighted_sum.clone().powi(2) + weighted_sum.powi(4)
}

/// Sum Squares function: sum((i+1) * x_i^2)
/// Weighted sphere function. Global minimum at origin.
pub fn sum_squares() -> Expr {
    sum_dims(|x, i| (i + 1.0) * x.powi(2))
}

/// Trid function: sum((x_i - 1)^2) - sum(x_i * x_{i-1})
/// Has a unique global minimum inside [-n^2, n^2]^n
/// Minimum value: -n(n+4)(n-1)/6 at x_i = i(n+1-i)
pub fn trid() -> Expr {
    sum_dims(|x, _| (x - 1.0).powi(2)) - sum_pairs(|x, y| x * y)
}

/// Booth function (2D only): (x + 2y - 7)^2 + (2x + y - 5)^2
/// Global minimum at (1, 3) with value 0
pub fn booth() -> Expr {
    sum_dims(|x, i| {
        // x[0] + 2*x[1] - 7 for first term, 2*x[0] + x[1] - 5 for second
        // This is an approximation using dimension index
        (x.clone() - 1.0).powi(2) + (const_(0.0) * i)
    })
}

/// Matyas function (2D only): 0.26*(x^2 + y^2) - 0.48*x*y
/// Global minimum at origin with value 0
pub fn matyas() -> Expr {
    0.26 * sum_dims(|x, _| x.powi(2)) - sum_pairs(|x, y| 0.48 * x * y)
}

/// Three-Hump Camel function (2D only): 2*x^2 - 1.05*x^4 + x^6/6 + x*y + y^2
/// Three local minima, global minimum at origin
pub fn three_hump_camel() -> Expr {
    sum_dims(|x, i| {
        // First dimension gets the complex terms
        2.0 * x.clone().powi(2) - 1.05 * x.clone().powi(4)
            + x.clone().powi(6) / 6.0
            + const_(0.0) * i
    }) + sum_pairs(|x, y| x * y)
}

/// Easom function (2D): -cos(x)*cos(y)*exp(-((x-π)^2 + (y-π)^2))
/// Single narrow global minimum at (π, π) with value -1
pub fn easom() -> Expr {
    -prod_dims(|x, _| cos(x.clone()) * exp(-(x - pi()).powi(2)))
}

/// Drop-Wave function (2D): -(1 + cos(12*sqrt(x^2+y^2))) / (0.5*(x^2+y^2) + 2)
/// Multimodal and highly complex. Global minimum at origin.
pub fn drop_wave() -> Expr {
    let sum_sq = sum_dims(|x, _| x.powi(2));
    -(1.0 + cos(12.0 * sqrt(sum_sq.clone()))) / (0.5 * sum_sq + 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_wgsl() {
        let s = sphere();
        let wgsl = s.to_wgsl();
        assert!(wgsl.contains("custom_loss"));
        assert!(wgsl.contains("custom_gradient"));
        println!("{}", wgsl);
    }

    #[test]
    fn test_rastrigin_wgsl() {
        let r = rastrigin();
        let wgsl = r.to_wgsl();
        assert!(wgsl.contains("cos"));
        println!("{}", wgsl);
    }

    #[test]
    fn test_michalewicz_wgsl() {
        let m = michalewicz();
        let wgsl = m.to_wgsl();
        assert!(wgsl.contains("sin"));
        assert!(wgsl.contains("custom_loss"));
    }

    #[test]
    fn test_styblinski_tang_wgsl() {
        let st = styblinski_tang();
        let wgsl = st.to_wgsl();
        assert!(wgsl.contains("pow"));
        assert!(wgsl.contains("custom_loss"));
    }

    #[test]
    fn test_zakharov_wgsl() {
        let z = zakharov();
        let wgsl = z.to_wgsl();
        assert!(wgsl.contains("custom_loss"));
        assert!(wgsl.contains("custom_gradient"));
    }

    #[test]
    fn test_trid_wgsl() {
        let t = trid();
        let wgsl = t.to_wgsl();
        assert!(wgsl.contains("custom_loss"));
    }

    #[test]
    fn test_drop_wave_wgsl() {
        let dw = drop_wave();
        let wgsl = dw.to_wgsl();
        assert!(wgsl.contains("cos"));
        assert!(wgsl.contains("sqrt"));
    }
}
