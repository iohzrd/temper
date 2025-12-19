//! Image Inpainting Demo
//!
//! Demonstrates thermodynamic image inpainting using 32x32 RGB images.
//! Inpainting fills in masked regions while preserving known pixels.
//!
//! Run with: cargo run --release --features gpu --bin image-inpainting

use temper::ThermodynamicSystem;
use temper::expr::custom_wgsl;

const IMG_SIZE: usize = 32;
const CHANNEL_DIM: usize = IMG_SIZE * IMG_SIZE; // 1024 per channel
const RGB_DIM: usize = CHANNEL_DIM * 3; // 3072 total

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    IMAGE INPAINTING DEMO                                 ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  32x32 RGB images with thermodynamic inpainting                          ║");
    println!("║  Each particle = one color image (3072 dimensions = 1024×3 channels)     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Demo 1: Generate a gradient image (no mask)
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("DEMO 1: 32x32 Rainbow Gradient (full image generation)");
    println!("═══════════════════════════════════════════════════════════════════════════\n");
    run_diffusion("rainbow32", generate_rainbow32_wgsl(), 20);

    // Demo 2: Inpaint center region
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("DEMO 2: Inpainting - Fill center square");
    println!("═══════════════════════════════════════════════════════════════════════════\n");
    run_inpainting("inpaint_center", 20);

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                              KEY INSIGHTS                                ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                          ║");
    println!("║  1. LARGER IMAGES: 32x32 RGB = 3072 dimensions (3x previous)             ║");
    println!("║                                                                          ║");
    println!("║  2. INPAINTING: Known pixels penalized if changed, unknown are free      ║");
    println!("║                                                                          ║");
    println!("║  3. SAME PHYSICS: Temperature annealing fills in plausible content       ║");
    println!("║                                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

fn run_diffusion(name: &str, loss_expr: temper::expr::Expr, n_particles: usize) {
    let mut system = ThermodynamicSystem::with_expr(n_particles, RGB_DIM, 1.0, loss_expr);

    println!("Target pattern: {}", name);
    println!(
        "Particles: {} (each is a 32x32 RGB image = 3072 dims)",
        n_particles
    );
    println!();

    // Show target pattern
    println!("Target image:");
    print_target_pattern_rgb(name);
    println!();

    // Annealing schedule
    let total_steps = 5000;
    let t_start = 1.0_f32;
    let t_end = 0.001_f32;

    let checkpoints = [0, 1000, 2000, 3000, 4000, 5000];

    for step in 0..=total_steps {
        let progress = step as f32 / total_steps as f32;
        let temp = t_start * (t_end / t_start).powf(progress);
        system.set_temperature(temp);

        if step > 0 {
            system.step();
        }

        if checkpoints.contains(&step) {
            let particles = system.read_particles();
            let (best_idx, best_energy) = particles
                .iter()
                .enumerate()
                .filter(|(_, p)| p.energy.is_finite())
                .min_by(|(_, a), (_, b)| a.energy.partial_cmp(&b.energy).unwrap())
                .map(|(i, p)| (i, p.energy))
                .unwrap_or((0, f32::MAX));

            println!(
                "Step {:>4} | T={:.4} | Energy={:.4}",
                step, temp, best_energy
            );

            if step == total_steps {
                let best = &particles[best_idx];
                let mut r = vec![0.0f32; CHANNEL_DIM];
                let mut g = vec![0.0f32; CHANNEL_DIM];
                let mut b = vec![0.0f32; CHANNEL_DIM];

                for i in 0..CHANNEL_DIM {
                    r[i] = best.pos[i].to_f32().clamp(0.0, 1.0);
                    g[i] = best.pos[CHANNEL_DIM + i].to_f32().clamp(0.0, 1.0);
                    b[i] = best.pos[2 * CHANNEL_DIM + i].to_f32().clamp(0.0, 1.0);
                }

                println!("\nFinal result:");
                print_rgb_image(&r, &g, &b);
            }
        }
    }
}

fn run_inpainting(name: &str, n_particles: usize) {
    // Create inpainting loss: penalize changes to known pixels, allow unknown
    let loss_expr = generate_inpainting_wgsl();
    let mut system = ThermodynamicSystem::with_expr(n_particles, RGB_DIM, 1.0, loss_expr);

    println!("Pattern: {} (inpainting center region)", name);
    println!(
        "Particles: {} (each is a 32x32 RGB image = 3072 dims)",
        n_particles
    );
    println!();

    // Show original with mask
    println!("Original image with masked center (to be filled):");
    print_masked_image();
    println!();

    // Annealing schedule
    let total_steps = 5000;
    let t_start = 1.0_f32;
    let t_end = 0.001_f32;

    let checkpoints = [0, 1000, 2000, 3000, 4000, 5000];

    for step in 0..=total_steps {
        let progress = step as f32 / total_steps as f32;
        let temp = t_start * (t_end / t_start).powf(progress);
        system.set_temperature(temp);

        if step > 0 {
            system.step();
        }

        if checkpoints.contains(&step) {
            let particles = system.read_particles();
            let (best_idx, best_energy) = particles
                .iter()
                .enumerate()
                .filter(|(_, p)| p.energy.is_finite())
                .min_by(|(_, a), (_, b)| a.energy.partial_cmp(&b.energy).unwrap())
                .map(|(i, p)| (i, p.energy))
                .unwrap_or((0, f32::MAX));

            println!(
                "Step {:>4} | T={:.4} | Energy={:.4}",
                step, temp, best_energy
            );

            if step == total_steps {
                let best = &particles[best_idx];
                let mut r = vec![0.0f32; CHANNEL_DIM];
                let mut g = vec![0.0f32; CHANNEL_DIM];
                let mut b = vec![0.0f32; CHANNEL_DIM];

                for i in 0..CHANNEL_DIM {
                    r[i] = best.pos[i].to_f32().clamp(0.0, 1.0);
                    g[i] = best.pos[CHANNEL_DIM + i].to_f32().clamp(0.0, 1.0);
                    b[i] = best.pos[2 * CHANNEL_DIM + i].to_f32().clamp(0.0, 1.0);
                }

                println!("\nInpainted result:");
                print_rgb_image(&r, &g, &b);
            }
        }
    }
}

/// Print RGB image using ANSI true color escape codes (subsampled for terminal)
fn print_rgb_image(r: &[f32], g: &[f32], b: &[f32]) {
    // Subsample 32x32 -> 16x16 for reasonable terminal output
    for y in (0..IMG_SIZE).step_by(2) {
        for x in (0..IMG_SIZE).step_by(2) {
            let idx = y * IMG_SIZE + x;
            let ri = (r[idx] * 255.0) as u8;
            let gi = (g[idx] * 255.0) as u8;
            let bi = (b[idx] * 255.0) as u8;
            print!("\x1b[48;2;{};{};{}m  \x1b[0m", ri, gi, bi);
        }
        println!();
    }
}

/// Print target pattern for reference
fn print_target_pattern_rgb(name: &str) {
    let (r, g, b) = match name {
        "rainbow32" => generate_rainbow32_target(),
        _ => (
            vec![0.5; CHANNEL_DIM],
            vec![0.5; CHANNEL_DIM],
            vec![0.5; CHANNEL_DIM],
        ),
    };
    print_rgb_image(&r, &g, &b);
}

/// Print masked image showing which regions will be filled
fn print_masked_image() {
    let (r, g, b) = generate_gradient_with_mask();
    // Show mask region as gray
    for y in (0..IMG_SIZE).step_by(2) {
        for x in (0..IMG_SIZE).step_by(2) {
            let idx = y * IMG_SIZE + x;
            // Check if in mask region (center 12x12)
            let in_mask = x >= 10 && x < 22 && y >= 10 && y < 22;
            if in_mask {
                // Gray for unknown
                print!("\x1b[48;2;128;128;128m??\x1b[0m");
            } else {
                let ri = (r[idx] * 255.0) as u8;
                let gi = (g[idx] * 255.0) as u8;
                let bi = (b[idx] * 255.0) as u8;
                print!("\x1b[48;2;{};{};{}m  \x1b[0m", ri, gi, bi);
            }
        }
        println!();
    }
}

// =============================================================================
// Target Pattern Generators (Rust)
// =============================================================================

fn generate_rainbow32_target() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = vec![0.0; CHANNEL_DIM];
    let mut g = vec![0.0; CHANNEL_DIM];
    let mut b = vec![0.0; CHANNEL_DIM];

    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let idx = y * IMG_SIZE + x;
            let hue = x as f32 / IMG_SIZE as f32;
            let (ri, gi, bi) = hsv_to_rgb(hue, 1.0, 1.0);
            r[idx] = ri;
            g[idx] = gi;
            b[idx] = bi;
        }
    }
    (r, g, b)
}

fn generate_gradient_with_mask() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Diagonal gradient for inpainting demo
    let mut r = vec![0.0; CHANNEL_DIM];
    let mut g = vec![0.0; CHANNEL_DIM];
    let mut b = vec![0.0; CHANNEL_DIM];

    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let idx = y * IMG_SIZE + x;
            let t = (x + y) as f32 / (2.0 * IMG_SIZE as f32);
            r[idx] = t;
            g[idx] = 1.0 - t;
            b[idx] = 0.5;
        }
    }
    (r, g, b)
}

/// HSV to RGB conversion (h, s, v all in [0, 1])
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = h * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

// =============================================================================
// WGSL Loss Function Generators
// =============================================================================

/// 32x32 rainbow gradient
fn generate_rainbow32_wgsl() -> temper::expr::Expr {
    let loss_code = r#"
fn hsv_to_r(h: f32) -> f32 {
    let h6 = h * 6.0;
    let i = floor(h6);
    let f = h6 - i;
    let ii = i32(i) % 6;
    if ii == 0 { return 1.0; }
    if ii == 1 { return 1.0 - f; }
    if ii == 2 { return 0.0; }
    if ii == 3 { return 0.0; }
    if ii == 4 { return f; }
    return 1.0;
}

fn hsv_to_g(h: f32) -> f32 {
    let h6 = h * 6.0;
    let i = floor(h6);
    let f = h6 - i;
    let ii = i32(i) % 6;
    if ii == 0 { return f; }
    if ii == 1 { return 1.0; }
    if ii == 2 { return 1.0; }
    if ii == 3 { return 1.0 - f; }
    if ii == 4 { return 0.0; }
    return 0.0;
}

fn hsv_to_b(h: f32) -> f32 {
    let h6 = h * 6.0;
    let i = floor(h6);
    let f = h6 - i;
    let ii = i32(i) % 6;
    if ii == 0 { return 0.0; }
    if ii == 1 { return 0.0; }
    if ii == 2 { return f; }
    if ii == 3 { return 1.0; }
    if ii == 4 { return 1.0; }
    return 1.0 - f;
}

fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    let size = 32u;
    var mse = 0.0;
    for (var i = 0u; i < 1024u; i = i + 1u) {
        let x = i % size;
        let hue = f32(x) / f32(size);

        let goal_r = hsv_to_r(hue);
        let goal_g = hsv_to_g(hue);
        let goal_b = hsv_to_b(hue);

        let diff_r = pos[i] - goal_r;
        let diff_g = pos[i + 1024u] - goal_g;
        let diff_b = pos[i + 2048u] - goal_b;
        mse = mse + diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
    }
    return mse / 3072.0;
}
"#;

    let grad_code = r#"
fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    if d_idx >= 3072u {
        return 0.0;
    }

    let size = 32u;
    let channel = d_idx / 1024u;
    let pixel_idx = d_idx % 1024u;
    let x = pixel_idx % size;
    let hue = f32(x) / f32(size);

    var goal = 0.0;
    if channel == 0u {
        goal = hsv_to_r(hue);
    } else if channel == 1u {
        goal = hsv_to_g(hue);
    } else {
        goal = hsv_to_b(hue);
    }

    return 2.0 * (pos[d_idx] - goal);
}
"#;

    custom_wgsl(loss_code, grad_code)
}

/// Inpainting loss: known pixels constrained, center region free with smoothness
fn generate_inpainting_wgsl() -> temper::expr::Expr {
    let loss_code = r#"
fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    let size = 32u;
    var loss = 0.0;

    // Known pixels loss (diagonal gradient)
    for (var i = 0u; i < 1024u; i = i + 1u) {
        let y = i / size;
        let x = i % size;

        // Mask: center 12x12 is unknown
        let in_mask = x >= 10u && x < 22u && y >= 10u && y < 22u;

        if !in_mask {
            // Known pixel - constrain to target gradient
            let t = f32(x + y) / 64.0;
            let goal_r = t;
            let goal_g = 1.0 - t;
            let goal_b = 0.5;

            let diff_r = pos[i] - goal_r;
            let diff_g = pos[i + 1024u] - goal_g;
            let diff_b = pos[i + 2048u] - goal_b;
            loss = loss + 10.0 * (diff_r * diff_r + diff_g * diff_g + diff_b * diff_b);
        }
    }

    // Smoothness loss for masked region (encourage coherent fill)
    for (var i = 0u; i < 1024u; i = i + 1u) {
        let y = i / size;
        let x = i % size;

        if x < size - 1u {
            // Horizontal smoothness
            let diff_r = pos[i] - pos[i + 1u];
            let diff_g = pos[i + 1024u] - pos[i + 1025u];
            let diff_b = pos[i + 2048u] - pos[i + 2049u];
            loss = loss + 0.1 * (diff_r * diff_r + diff_g * diff_g + diff_b * diff_b);
        }
        if y < size - 1u {
            // Vertical smoothness
            let next_row = i + size;
            let diff_r = pos[i] - pos[next_row];
            let diff_g = pos[i + 1024u] - pos[next_row + 1024u];
            let diff_b = pos[i + 2048u] - pos[next_row + 2048u];
            loss = loss + 0.1 * (diff_r * diff_r + diff_g * diff_g + diff_b * diff_b);
        }
    }

    return loss / 3072.0;
}
"#;

    let grad_code = r#"
fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    if d_idx >= 3072u {
        return 0.0;
    }

    let size = 32u;
    let channel = d_idx / 1024u;
    let pixel_idx = d_idx % 1024u;
    let y = pixel_idx / size;
    let x = pixel_idx % size;

    var grad = 0.0;

    // Mask: center 12x12 is unknown
    let in_mask = x >= 10u && x < 22u && y >= 10u && y < 22u;

    if !in_mask {
        // Known pixel gradient
        let t = f32(x + y) / 64.0;
        var goal = 0.0;
        if channel == 0u { goal = t; }
        else if channel == 1u { goal = 1.0 - t; }
        else { goal = 0.5; }

        grad = grad + 20.0 * (pos[d_idx] - goal);
    }

    // Smoothness gradients
    let base_idx = channel * 1024u + pixel_idx;

    // Current pixel contributes to smoothness with neighbors
    if x > 0u {
        grad = grad + 0.2 * (pos[base_idx] - pos[base_idx - 1u]);
    }
    if x < size - 1u {
        grad = grad + 0.2 * (pos[base_idx] - pos[base_idx + 1u]);
    }
    if y > 0u {
        grad = grad + 0.2 * (pos[base_idx] - pos[base_idx - size]);
    }
    if y < size - 1u {
        grad = grad + 0.2 * (pos[base_idx] - pos[base_idx + size]);
    }

    return grad / 3072.0;
}
"#;

    custom_wgsl(loss_code, grad_code)
}
