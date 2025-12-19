// MNIST PCA Classification Loss Function
// Network: 8 inputs (PCA features) -> 4 hidden (tanh) -> 5 outputs (softmax)
// Parameters (61 total):
//   pos[0-31]:  W1 (4x8 input->hidden weights, neuron-major)
//   pos[32-35]: b1 (4 hidden biases)
//   pos[36-55]: W2 (5x4 hidden->output weights, output-major)
//   pos[56-60]: b2 (5 output biases)

// Synthetic PCA-like features for digits 0-4 (30 samples per digit = 150 total)
// Each digit has distinct patterns in different PCA components
const MNIST_SIZE: u32 = 150u;
const SAMPLES_PER_CLASS: u32 = 30u;

// Generate deterministic synthetic feature for a sample
fn mnist_feature(sample_idx: u32, feature_idx: u32) -> f32 {
    let digit = sample_idx / SAMPLES_PER_CLASS;
    let within_class = sample_idx % SAMPLES_PER_CLASS;

    // Base pattern for each digit (digit, feature) -> base value
    var base = 0.3;
    if digit == 0u {
        if feature_idx == 0u { base = 0.8; }
        else if feature_idx == 1u { base = 0.2; }
        else if feature_idx == 2u { base = 0.7; }
    } else if digit == 1u {
        if feature_idx == 0u { base = 0.2; }
        else if feature_idx == 1u { base = 0.9; }
        else if feature_idx == 3u { base = 0.6; }
    } else if digit == 2u {
        if feature_idx == 0u { base = 0.5; }
        else if feature_idx == 2u { base = 0.3; }
        else if feature_idx == 4u { base = 0.8; }
    } else if digit == 3u {
        if feature_idx == 1u { base = 0.4; }
        else if feature_idx == 3u { base = 0.7; }
        else if feature_idx == 5u { base = 0.6; }
    } else if digit == 4u {
        if feature_idx == 2u { base = 0.6; }
        else if feature_idx == 4u { base = 0.4; }
        else if feature_idx == 6u { base = 0.7; }
    }

    // Deterministic noise based on sample and feature
    let seed = sample_idx * 8u + feature_idx + 54321u;
    let noise_val = f32((seed ^ (seed >> 7u) ^ (seed << 3u)) % 1000u) / 5000.0 - 0.1;

    return clamp(base + noise_val, 0.0, 1.0);
}

fn mnist_label(sample_idx: u32) -> u32 {
    return sample_idx / SAMPLES_PER_CLASS;
}

fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    var total_loss = 0.0;

    for (var i = 0u; i < MNIST_SIZE; i = i + 1u) {
        // Get input features
        let x0 = mnist_feature(i, 0u);
        let x1 = mnist_feature(i, 1u);
        let x2 = mnist_feature(i, 2u);
        let x3 = mnist_feature(i, 3u);
        let x4 = mnist_feature(i, 4u);
        let x5 = mnist_feature(i, 5u);
        let x6 = mnist_feature(i, 6u);
        let x7 = mnist_feature(i, 7u);

        // Layer 1: 8 -> 4 (tanh)
        let h0 = tanh(pos[0] * x0 + pos[1] * x1 + pos[2] * x2 + pos[3] * x3 +
                      pos[4] * x4 + pos[5] * x5 + pos[6] * x6 + pos[7] * x7 + pos[32]);
        let h1 = tanh(pos[8] * x0 + pos[9] * x1 + pos[10] * x2 + pos[11] * x3 +
                      pos[12] * x4 + pos[13] * x5 + pos[14] * x6 + pos[15] * x7 + pos[33]);
        let h2 = tanh(pos[16] * x0 + pos[17] * x1 + pos[18] * x2 + pos[19] * x3 +
                      pos[20] * x4 + pos[21] * x5 + pos[22] * x6 + pos[23] * x7 + pos[34]);
        let h3 = tanh(pos[24] * x0 + pos[25] * x1 + pos[26] * x2 + pos[27] * x3 +
                      pos[28] * x4 + pos[29] * x5 + pos[30] * x6 + pos[31] * x7 + pos[35]);

        // Layer 2: 4 -> 5 (logits)
        let z0 = pos[36] * h0 + pos[37] * h1 + pos[38] * h2 + pos[39] * h3 + pos[56];
        let z1 = pos[40] * h0 + pos[41] * h1 + pos[42] * h2 + pos[43] * h3 + pos[57];
        let z2 = pos[44] * h0 + pos[45] * h1 + pos[46] * h2 + pos[47] * h3 + pos[58];
        let z3 = pos[48] * h0 + pos[49] * h1 + pos[50] * h2 + pos[51] * h3 + pos[59];
        let z4 = pos[52] * h0 + pos[53] * h1 + pos[54] * h2 + pos[55] * h3 + pos[60];

        // Stable softmax + cross-entropy
        let max_z = max(max(max(max(z0, z1), z2), z3), z4);
        let e0 = exp(z0 - max_z);
        let e1 = exp(z1 - max_z);
        let e2 = exp(z2 - max_z);
        let e3 = exp(z3 - max_z);
        let e4 = exp(z4 - max_z);
        let sum_e = e0 + e1 + e2 + e3 + e4;

        // Cross-entropy loss
        let label = mnist_label(i);
        var log_prob = 0.0;
        if label == 0u {
            log_prob = log(e0 / sum_e + 0.0001);
        } else if label == 1u {
            log_prob = log(e1 / sum_e + 0.0001);
        } else if label == 2u {
            log_prob = log(e2 / sum_e + 0.0001);
        } else if label == 3u {
            log_prob = log(e3 / sum_e + 0.0001);
        } else {
            log_prob = log(e4 / sum_e + 0.0001);
        }
        total_loss = total_loss - log_prob;
    }

    return total_loss / f32(MNIST_SIZE);
}
