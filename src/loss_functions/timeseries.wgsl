// Time Series Forecasting Loss Function
// Network: 8 inputs (window) -> 4 hidden (tanh) -> 1 output (sigmoid)
// Parameters (41 total):
//   pos[0-31]:  W1 (4x8 input->hidden weights, neuron-major)
//   pos[32-35]: b1 (4 hidden biases)
//   pos[36-39]: W2 (4x1 hidden->output weights)
//   pos[40]:    b2 (1 output bias)

// Pre-computed sine wave with noise (192 windows of size 8, with targets)
// Generated from: (sin(t) + noise + 1) / 2, t = 0.1 * i
const TS_SIZE: u32 = 64u;  // Use subset for faster training

// Sine wave values (normalized to [0,1])
// sin(0.0*i) + 1) / 2 for i = 0..200
fn sine_value(idx: u32) -> f32 {
    let t = f32(idx) * 0.1;
    // Deterministic "noise" based on index
    let noise_seed = idx * 12345u + 7919u;
    let noise = f32((noise_seed ^ (noise_seed >> 5u)) % 1000u) / 10000.0 - 0.05;
    return (sin(t) + 1.0) / 2.0 + noise;
}

// sigmoid is already defined in the base shader

fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    var total_loss = 0.0;

    // Iterate through windows
    for (var w = 0u; w < TS_SIZE; w = w + 1u) {
        // Get window of 8 values
        let x0 = sine_value(w);
        let x1 = sine_value(w + 1u);
        let x2 = sine_value(w + 2u);
        let x3 = sine_value(w + 3u);
        let x4 = sine_value(w + 4u);
        let x5 = sine_value(w + 5u);
        let x6 = sine_value(w + 6u);
        let x7 = sine_value(w + 7u);
        let y_target = sine_value(w + 8u);

        // Layer 1: 8 -> 4 (tanh)
        let h0 = tanh(pos[0] * x0 + pos[1] * x1 + pos[2] * x2 + pos[3] * x3 +
                      pos[4] * x4 + pos[5] * x5 + pos[6] * x6 + pos[7] * x7 + pos[32]);
        let h1 = tanh(pos[8] * x0 + pos[9] * x1 + pos[10] * x2 + pos[11] * x3 +
                      pos[12] * x4 + pos[13] * x5 + pos[14] * x6 + pos[15] * x7 + pos[33]);
        let h2 = tanh(pos[16] * x0 + pos[17] * x1 + pos[18] * x2 + pos[19] * x3 +
                      pos[20] * x4 + pos[21] * x5 + pos[22] * x6 + pos[23] * x7 + pos[34]);
        let h3 = tanh(pos[24] * x0 + pos[25] * x1 + pos[26] * x2 + pos[27] * x3 +
                      pos[28] * x4 + pos[29] * x5 + pos[30] * x6 + pos[31] * x7 + pos[35]);

        // Layer 2: 4 -> 1 (sigmoid)
        let out = sigmoid(pos[36] * h0 + pos[37] * h1 + pos[38] * h2 + pos[39] * h3 + pos[40]);

        // MSE loss
        let err = out - y_target;
        total_loss = total_loss + err * err;
    }

    return total_loss / f32(TS_SIZE);
}
