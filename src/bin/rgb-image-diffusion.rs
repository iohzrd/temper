//! RGB Image Diffusion Demo
//!
//! Demonstrates thermodynamic color image generation using 16x16 RGB images.
//! Each particle represents a 768-dimensional image (16×16×3) that "denoises"
//! from pure noise into colorful patterns through temperature annealing.
//!
//! Layout: R,G,B planar - first 256 dims are R, next 256 are G, last 256 are B
//!
//! Run with: cargo run --release --features gpu --bin rgb-image-diffusion

use temper::ThermodynamicSystem;
use temper::expr::custom_wgsl;

const IMG_SIZE: usize = 16;
const CHANNEL_DIM: usize = IMG_SIZE * IMG_SIZE; // 256 per channel
const RGB_DIM: usize = CHANNEL_DIM * 3; // 768 total

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    RGB IMAGE DIFFUSION DEMO                              ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  16x16 RGB color images generated via thermodynamic denoising            ║");
    println!("║  Each particle = one color image (768 dimensions = 256×3 channels)       ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Demo 1: Red Circle on Blue Background
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("DEMO 1: Red Circle on Blue Background");
    println!("═══════════════════════════════════════════════════════════════════════════\n");
    run_rgb_diffusion("red_circle", generate_red_circle_wgsl(), 50);

    // Demo 2: Rainbow Gradient
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("DEMO 2: Rainbow Horizontal Gradient");
    println!("═══════════════════════════════════════════════════════════════════════════\n");
    run_rgb_diffusion("rainbow", generate_rainbow_wgsl(), 50);

    // Demo 3: Color Checkerboard (Red/Green)
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("DEMO 3: Red/Green Checkerboard");
    println!("═══════════════════════════════════════════════════════════════════════════\n");
    run_rgb_diffusion("checkerboard", generate_color_checkerboard_wgsl(), 50);

    // Demo 4: Concentric Color Rings
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("DEMO 4: Concentric Color Rings");
    println!("═══════════════════════════════════════════════════════════════════════════\n");
    run_rgb_diffusion("rings", generate_color_rings_wgsl(), 50);

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                              KEY INSIGHTS                                ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                          ║");
    println!("║  1. COLOR IMAGES: 768 dimensions = 16×16 pixels × 3 RGB channels         ║");
    println!("║                                                                          ║");
    println!("║  2. PLANAR LAYOUT: pos[0..256]=R, pos[256..256]=G, pos[256..768]=B       ║");
    println!("║                                                                          ║");
    println!("║  3. SAME PHYSICS: Temperature annealing works identically for colors    ║");
    println!("║                                                                          ║");
    println!("║  4. SCALABILITY: MAX_DIMENSIONS=1024 supports even larger images!        ║");
    println!("║                                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

fn run_rgb_diffusion(name: &str, loss_expr: temper::expr::Expr, n_particles: usize) {
    let mut system = ThermodynamicSystem::with_expr(n_particles, RGB_DIM, 1.0, loss_expr);

    println!("Target pattern: {}", name);
    println!(
        "Particles: {} (each is a 16x16 RGB image = 768 dims)",
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

            if step == total_steps || step % 2500 == 0 {
                // Extract and display RGB image
                let best = &particles[best_idx];
                let mut r = vec![0.0f32; CHANNEL_DIM];
                let mut g = vec![0.0f32; CHANNEL_DIM];
                let mut b = vec![0.0f32; CHANNEL_DIM];

                for i in 0..CHANNEL_DIM {
                    r[i] = best.pos[i].to_f32().clamp(0.0, 1.0);
                    g[i] = best.pos[CHANNEL_DIM + i].to_f32().clamp(0.0, 1.0);
                    b[i] = best.pos[2 * CHANNEL_DIM + i].to_f32().clamp(0.0, 1.0);
                }

                println!();
                print_rgb_image(&r, &g, &b);
                println!();
            }
        }
    }
}

/// Print RGB image using ANSI true color escape codes
fn print_rgb_image(r: &[f32], g: &[f32], b: &[f32]) {
    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let idx = y * IMG_SIZE + x;
            let ri = (r[idx] * 255.0) as u8;
            let gi = (g[idx] * 255.0) as u8;
            let bi = (b[idx] * 255.0) as u8;
            // Use ANSI true color: \x1b[48;2;R;G;Bm for background color
            print!("\x1b[48;2;{};{};{}m  \x1b[0m", ri, gi, bi);
        }
        println!();
    }
}

/// Print target pattern for reference
fn print_target_pattern_rgb(name: &str) {
    let (r, g, b) = match name {
        "red_circle" => generate_red_circle_target(),
        "rainbow" => generate_rainbow_target(),
        "checkerboard" => generate_color_checkerboard_target(),
        "rings" => generate_color_rings_target(),
        _ => (
            vec![0.5; CHANNEL_DIM],
            vec![0.5; CHANNEL_DIM],
            vec![0.5; CHANNEL_DIM],
        ),
    };
    print_rgb_image(&r, &g, &b);
}

// =============================================================================
// Target Pattern Generators (Rust)
// =============================================================================

fn generate_red_circle_target() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = vec![0.0; CHANNEL_DIM];
    let mut g = vec![0.0; CHANNEL_DIM];
    let mut b = vec![0.0; CHANNEL_DIM];

    let center = IMG_SIZE as f32 / 2.0;
    let radius = IMG_SIZE as f32 / 3.0;

    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let idx = y * IMG_SIZE + x;
            let dx = x as f32 + 0.5 - center;
            let dy = y as f32 + 0.5 - center;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < radius {
                // Red circle
                r[idx] = 1.0;
                g[idx] = 0.0;
                b[idx] = 0.0;
            } else {
                // Blue background
                r[idx] = 0.0;
                g[idx] = 0.0;
                b[idx] = 0.8;
            }
        }
    }
    (r, g, b)
}

fn generate_rainbow_target() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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

fn generate_color_checkerboard_target() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = vec![0.0; CHANNEL_DIM];
    let mut g = vec![0.0; CHANNEL_DIM];
    let mut b = vec![0.0; CHANNEL_DIM];

    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let idx = y * IMG_SIZE + x;
            let cell_x = x / 4;
            let cell_y = y / 4;
            if (cell_x + cell_y) % 2 == 0 {
                // Red
                r[idx] = 1.0;
                g[idx] = 0.0;
                b[idx] = 0.0;
            } else {
                // Green
                r[idx] = 0.0;
                g[idx] = 1.0;
                b[idx] = 0.0;
            }
        }
    }
    (r, g, b)
}

fn generate_color_rings_target() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = vec![0.0; CHANNEL_DIM];
    let mut g = vec![0.0; CHANNEL_DIM];
    let mut b = vec![0.0; CHANNEL_DIM];

    let center = IMG_SIZE as f32 / 2.0;

    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let idx = y * IMG_SIZE + x;
            let dx = x as f32 + 0.5 - center;
            let dy = y as f32 + 0.5 - center;
            let dist = (dx * dx + dy * dy).sqrt();

            // Color based on distance - creates rings
            let hue = (dist / (IMG_SIZE as f32 / 2.0)).fract();
            let (ri, gi, bi) = hsv_to_rgb(hue, 1.0, 1.0);
            r[idx] = ri;
            g[idx] = gi;
            b[idx] = bi;
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

/// Generate WGSL for red circle on blue background
fn generate_red_circle_wgsl() -> temper::expr::Expr {
    let loss_code = r#"
fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    let size = 16.0;
    let center = size / 2.0;
    let radius = size / 3.0;

    var mse = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let y = f32(i / 16u);
        let x = f32(i % 16u);

        let dx = x + 0.5 - center;
        let dy = y + 0.5 - center;
        let dist = sqrt(dx * dx + dy * dy);

        // Target RGB values
        var goal_r = 0.0;
        var goal_g = 0.0;
        var goal_b = 0.8;  // Blue background
        if dist < radius {
            goal_r = 1.0;   // Red circle
            goal_g = 0.0;
            goal_b = 0.0;
        }

        // MSE for each channel (planar layout)
        let diff_r = pos[i] - goal_r;
        let diff_g = pos[i + 256u] - goal_g;
        let diff_b = pos[i + 256u] - goal_b;
        mse = mse + diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
    }
    return mse / 768.0;
}
"#;

    let grad_code = r#"
fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    if d_idx >= 768u {
        return 0.0;
    }

    let size = 16.0;
    let center = size / 2.0;
    let radius = size / 3.0;

    // Determine which channel and pixel
    let channel = d_idx / 256u;
    let pixel_idx = d_idx % 256u;

    let y = f32(pixel_idx / 16u);
    let x = f32(pixel_idx % 16u);

    let dx = x + 0.5 - center;
    let dy = y + 0.5 - center;
    let dist = sqrt(dx * dx + dy * dy);

    // Goal for this channel
    var goal = 0.0;
    if channel == 0u {
        // Red channel
        if dist < radius { goal = 1.0; }
    } else if channel == 1u {
        // Green channel
        goal = 0.0;
    } else {
        // Blue channel
        if dist >= radius { goal = 0.8; }
    }

    return 2.0 * (pos[d_idx] - goal);
}
"#;

    custom_wgsl(loss_code, grad_code)
}

/// Generate WGSL for rainbow gradient
fn generate_rainbow_wgsl() -> temper::expr::Expr {
    let loss_code = r#"
// HSV to RGB in WGSL
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
    var mse = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let x = i % 16u;
        let hue = f32(x) / 16.0;

        let goal_r = hsv_to_r(hue);
        let goal_g = hsv_to_g(hue);
        let goal_b = hsv_to_b(hue);

        let diff_r = pos[i] - goal_r;
        let diff_g = pos[i + 256u] - goal_g;
        let diff_b = pos[i + 256u] - goal_b;
        mse = mse + diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
    }
    return mse / 768.0;
}
"#;

    let grad_code = r#"
fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    if d_idx >= 768u {
        return 0.0;
    }

    let channel = d_idx / 256u;
    let pixel_idx = d_idx % 256u;
    let x = pixel_idx % 16u;
    let hue = f32(x) / 16.0;

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

/// Generate WGSL for red/green checkerboard
fn generate_color_checkerboard_wgsl() -> temper::expr::Expr {
    let loss_code = r#"
fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    var mse = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let y = i / 16u;
        let x = i % 16u;

        let cell_x = x / 4u;
        let cell_y = y / 4u;

        var goal_r = 0.0;
        var goal_g = 1.0;  // Green
        var goal_b = 0.0;
        if (cell_x + cell_y) % 2u == 0u {
            goal_r = 1.0;  // Red
            goal_g = 0.0;
            goal_b = 0.0;
        }

        let diff_r = pos[i] - goal_r;
        let diff_g = pos[i + 256u] - goal_g;
        let diff_b = pos[i + 256u] - goal_b;
        mse = mse + diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
    }
    return mse / 768.0;
}
"#;

    let grad_code = r#"
fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    if d_idx >= 768u {
        return 0.0;
    }

    let channel = d_idx / 256u;
    let pixel_idx = d_idx % 256u;
    let y = pixel_idx / 16u;
    let x = pixel_idx % 16u;

    let cell_x = x / 4u;
    let cell_y = y / 4u;
    let is_red = (cell_x + cell_y) % 2u == 0u;

    var goal = 0.0;
    if channel == 0u {
        // Red channel
        if is_red { goal = 1.0; }
    } else if channel == 1u {
        // Green channel
        if !is_red { goal = 1.0; }
    }
    // Blue always 0

    return 2.0 * (pos[d_idx] - goal);
}
"#;

    custom_wgsl(loss_code, grad_code)
}

/// Generate WGSL for concentric color rings
fn generate_color_rings_wgsl() -> temper::expr::Expr {
    let loss_code = r#"
fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    let size = 16.0;
    let center = size / 2.0;

    var mse = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let y = f32(i / 16u);
        let x = f32(i % 16u);

        let dx = x + 0.5 - center;
        let dy = y + 0.5 - center;
        let dist = sqrt(dx * dx + dy * dy);

        let hue = fract(dist / (size / 2.0));
        let goal_r = hsv_to_r(hue);
        let goal_g = hsv_to_g(hue);
        let goal_b = hsv_to_b(hue);

        let diff_r = pos[i] - goal_r;
        let diff_g = pos[i + 256u] - goal_g;
        let diff_b = pos[i + 256u] - goal_b;
        mse = mse + diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
    }
    return mse / 768.0;
}
"#;

    let grad_code = r#"
fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    if d_idx >= 768u {
        return 0.0;
    }

    let size = 16.0;
    let center = size / 2.0;

    let channel = d_idx / 256u;
    let pixel_idx = d_idx % 256u;

    let y = f32(pixel_idx / 16u);
    let x = f32(pixel_idx % 16u);

    let dx = x + 0.5 - center;
    let dy = y + 0.5 - center;
    let dist = sqrt(dx * dx + dy * dy);

    let hue = fract(dist / (size / 2.0));

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
