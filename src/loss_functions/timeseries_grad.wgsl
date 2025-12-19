// Numerical gradient for Time series forecasting loss
fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    if d_idx >= 41u {
        return 0.0;
    }

    let eps = 0.001;
    var pos_plus = pos;
    var pos_minus = pos;
    pos_plus[d_idx] = pos[d_idx] + eps;
    pos_minus[d_idx] = pos[d_idx] - eps;
    return (custom_loss(pos_plus, dim) - custom_loss(pos_minus, dim)) / (2.0 * eps);
}
