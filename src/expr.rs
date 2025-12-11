//! Expression DSL for custom loss functions
//!
//! Build loss functions from composable primitives that compile to GPU-accelerated WGSL.
//!
//! # Example
//! ```ignore
//! use nbody_entropy::expr::*;
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
pub fn abs(e: Expr) -> Expr { Expr::Abs(Arc::new(e)) }
pub fn sin(e: Expr) -> Expr { Expr::Sin(Arc::new(e)) }
pub fn cos(e: Expr) -> Expr { Expr::Cos(Arc::new(e)) }
pub fn tan(e: Expr) -> Expr { Expr::Tan(Arc::new(e)) }
pub fn exp(e: Expr) -> Expr { Expr::Exp(Arc::new(e)) }
pub fn ln(e: Expr) -> Expr { Expr::Ln(Arc::new(e)) }
pub fn sqrt(e: Expr) -> Expr { Expr::Sqrt(Arc::new(e)) }
pub fn tanh(e: Expr) -> Expr { Expr::Tanh(Arc::new(e)) }
pub fn floor(e: Expr) -> Expr { Expr::Floor(Arc::new(e)) }
pub fn ceil(e: Expr) -> Expr { Expr::Ceil(Arc::new(e)) }
pub fn sign(e: Expr) -> Expr { Expr::Sign(Arc::new(e)) }

// Binary functions
pub fn min(a: Expr, b: Expr) -> Expr { Expr::Min(Arc::new(a), Arc::new(b)) }
pub fn max(a: Expr, b: Expr) -> Expr { Expr::Max(Arc::new(a), Arc::new(b)) }
pub fn pow(base: Expr, exp: Expr) -> Expr { Expr::Pow(Arc::new(base), Arc::new(exp)) }
pub fn modulo(a: Expr, b: Expr) -> Expr { Expr::Mod(Arc::new(a), Arc::new(b)) }

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
    /// Generate complete WGSL loss function
    pub fn to_wgsl(&self) -> String {
        let mut counter = 0;
        let mut helpers = Vec::new();
        let body = self.to_wgsl_with_helpers("x", "i", &mut counter, &mut helpers);

        let helpers_code = helpers.join("\n\n");

        format!(
            r#"{helpers_code}

fn custom_loss(pos: array<f32, 64>, dim: u32) -> f32 {{
    return {body};
}}

fn custom_gradient(pos: array<f32, 64>, dim: u32, d_idx: u32) -> f32 {{
    let eps = 0.001;
    var pos_plus = pos;
    var pos_minus = pos;
    pos_plus[d_idx] = pos[d_idx] + eps;
    pos_minus[d_idx] = pos[d_idx] - eps;
    return (custom_loss(pos_plus, dim) - custom_loss(pos_minus, dim)) / (2.0 * eps);
}}"#
        )
    }

    /// Generate WGSL code with helper functions for reductions
    fn to_wgsl_with_helpers(&self, var_name: &str, idx_name: &str, counter: &mut u32, helpers: &mut Vec<String>) -> String {
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

            Expr::Neg(e) => format!("(-{})", e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Abs(e) => format!("abs({})", e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Sin(e) => format!("sin({})", e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Cos(e) => format!("cos({})", e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Tan(e) => format!("tan({})", e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Exp(e) => format!("exp({})", e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Ln(e) => format!("log({})", e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Sqrt(e) => format!("sqrt({})", e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Tanh(e) => format!("tanh({})", e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Floor(e) => format!("floor({})", e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Ceil(e) => format!("ceil({})", e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Sign(e) => format!("sign({})", e.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),

            Expr::Add(a, b) => format!("({} + {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Sub(a, b) => format!("({} - {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Mul(a, b) => format!("({} * {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Div(a, b) => format!("({} / {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Pow(a, b) => format!("pow({}, {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Min(a, b) => format!("min({}, {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Max(a, b) => format!("max({}, {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),
            Expr::Mod(a, b) => format!("({} % {})",
                a.to_wgsl_with_helpers(var_name, idx_name, counter, helpers),
                b.to_wgsl_with_helpers(var_name, idx_name, counter, helpers)),

            Expr::SumDims(f) => {
                *counter += 1;
                let fn_name = format!("sum_helper_{}", counter);
                let loop_var = "i";
                let x_var = "x";

                let body_expr = f(Expr::Var, Expr::DimIndex);
                let mut inner_counter = 100 + *counter * 10;
                let body_code = body_expr.to_wgsl_with_helpers(x_var, loop_var, &mut inner_counter, helpers);

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
                let body_code = body_expr.to_wgsl_with_helpers(x_var, loop_var, &mut inner_counter, helpers);

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
                let body_code = body_expr.to_wgsl_with_helpers(x_var, loop_var, &mut inner_counter, helpers);

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
    10.0 * dim_count() + sum_dims(|x, _| {
        x.clone().powi(2) - 10.0 * cos(2.0 * pi() * x)
    })
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

    const_(-a) * exp(const_(-b) * sqrt(sum_sq / dim_count()))
        - exp(sum_cos / dim_count())
        + const_(a + std::f32::consts::E)
}

/// Griewank function: 1 + sum(x^2/4000) - prod(cos(x/sqrt(i+1))), minimum at origin
pub fn griewank() -> Expr {
    const_(1.0)
        + sum_dims(|x, _| x.powi(2) / 4000.0)
        - prod_dims(|x, i| cos(x / sqrt(i + 1.0)))
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
}
