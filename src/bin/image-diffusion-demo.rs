//! Image Diffusion Demo
//!
//! Demonstrates thermodynamic image generation using 16x16 grayscale images.
//! Each particle represents a 256-dimensional image that "denoises" from pure
//! noise into recognizable patterns through temperature annealing.
//!
//! This directly connects Temper's Langevin dynamics to image diffusion models:
//! - High T: Pure noise (like diffusion t=T)
//! - Low T: Clean samples from target distribution (like diffusion t=0)
//!
//! Run with: cargo run --release --features gpu --bin image-diffusion-demo

use temper::ThermodynamicSystem;
use temper::expr::custom_wgsl;

const IMG_SIZE: usize = 16;
const IMG_DIM: usize = IMG_SIZE * IMG_SIZE; // 256

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    IMAGE DIFFUSION DEMO                                  ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  16x16 grayscale images generated via thermodynamic denoising            ║");
    println!("║  Each particle = one image (256 dimensions)                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Demo 1: Simple Circle
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("DEMO 1: Generating a Circle");
    println!("═══════════════════════════════════════════════════════════════════════════\n");
    run_image_diffusion("circle", generate_circle_wgsl(), 100);

    // Demo 2: Cross/Plus pattern
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("DEMO 2: Generating a Cross");
    println!("═══════════════════════════════════════════════════════════════════════════\n");
    run_image_diffusion("cross", generate_cross_wgsl(), 100);

    // Demo 3: Checkerboard
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("DEMO 3: Generating a Checkerboard");
    println!("═══════════════════════════════════════════════════════════════════════════\n");
    run_image_diffusion("checkerboard", generate_checkerboard_wgsl(), 100);

    // Demo 4: Multi-pattern mixture (particles can converge to different patterns)
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("DEMO 4: Multi-Pattern Mixture (Circle OR Cross)");
    println!("═══════════════════════════════════════════════════════════════════════════\n");
    run_image_diffusion("mixture", generate_mixture_wgsl(), 200);

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                              KEY INSIGHTS                                ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                          ║");
    println!("║  1. IMAGES AS PARTICLES: Each 256-dim particle IS a 16x16 image          ║");
    println!("║                                                                          ║");
    println!("║  2. ENERGY = IMAGE PRIOR: MSE to target defines the \"data distribution\" ║");
    println!("║                                                                          ║");
    println!("║  3. DENOISING = ANNEALING: Cooling from T_high → T_low IS denoising      ║");
    println!("║                                                                          ║");
    println!("║  4. MULTIMODAL GENERATION: With mixture priors, different particles      ║");
    println!("║     converge to different valid images (diversity via SVGD repulsion)    ║");
    println!("║                                                                          ║");
    println!("║  5. REAL DIFFUSION MODELS: Same math, but learn the energy function      ║");
    println!("║     E(x) = -log p(x) from data instead of defining it analytically       ║");
    println!("║                                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

fn run_image_diffusion(name: &str, loss_expr: temper::expr::Expr, n_particles: usize) {
    // Use lower temperature for image generation (values should be in [0,1])
    let mut system = ThermodynamicSystem::with_expr(n_particles, IMG_DIM, 1.0, loss_expr);

    println!("Target pattern: {}", name);
    println!("Particles: {} (each is a 16x16 image)", n_particles);
    println!();

    // Show target pattern
    println!("Target image:");
    print_target_pattern(name);
    println!();

    // Annealing schedule - start lower, more aggressive cooling
    let total_steps = 5000;
    let t_start = 1.0_f32;
    let t_end = 0.001_f32;

    let checkpoints = [0, 1000, 2000, 3000, 4000, 5000];

    for step in 0..=total_steps {
        let progress = step as f32 / total_steps as f32;
        let temp = t_start * (t_end / t_start).powf(progress);
        system.set_temperature(temp);

        if step > 0 {
            system.step_leapfrog_only(); // Pure gradient descent, no Metropolis accept/reject
        }

        if checkpoints.contains(&step) {
            let particles = system.read_particles();

            // Find best particle (lowest energy)
            let best = particles
                .iter()
                .filter(|p| !p.energy.is_nan())
                .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

            if let Some(p) = best {
                let diffusion_t = ((1.0 - progress) * 1000.0) as u32;
                let phase = match diffusion_t {
                    800..=1000 => "Pure noise",
                    500..=799 => "Noisy",
                    200..=499 => "Emerging",
                    50..=199 => "Clear",
                    _ => "Final",
                };

                println!(
                    "Step {:4} | T={:.4} | diffusion_t={:4} | {} | energy={:.3}",
                    step, temp, diffusion_t, phase, p.energy
                );
                print_image(&p.pos, IMG_SIZE);
                println!();
            }
        }
    }

    // Final statistics
    let final_particles = system.read_particles();
    let energies: Vec<f32> = final_particles
        .iter()
        .filter(|p| !p.energy.is_nan())
        .map(|p| p.energy)
        .collect();

    let mean_energy: f32 = energies.iter().sum::<f32>() / energies.len() as f32;
    let min_energy = energies.iter().cloned().fold(f32::MAX, f32::min);

    println!("Final results:");
    println!("  Mean energy: {:.4}", mean_energy);
    println!("  Best energy: {:.4}", min_energy);
}

/// Print a 16x16 image using Unicode block characters
fn print_image(pos: &[half::f16], size: usize) {
    // Use 4 levels: space, light, medium, full block
    let chars = [' ', '░', '▒', '▓', '█'];

    for y in 0..size {
        print!("    ");
        for x in 0..size {
            let idx = y * size + x;
            // Clamp to [0, 1] range
            let val = pos[idx].to_f32().clamp(0.0, 1.0);
            // Map to character index (0-4)
            let char_idx = (val * 4.0).round() as usize;
            let char_idx = char_idx.min(4);
            print!("{}", chars[char_idx]);
        }
        println!();
    }
}

/// Print target pattern for reference
fn print_target_pattern(name: &str) {
    let pattern = match name {
        "circle" => generate_circle_pattern(),
        "cross" => generate_cross_pattern(),
        "checkerboard" => generate_checkerboard_pattern(),
        "mixture" => generate_circle_pattern(), // Show circle as example
        _ => vec![0.0; IMG_DIM],
    };

    let chars = [' ', '░', '▒', '▓', '█'];
    for y in 0..IMG_SIZE {
        print!("    ");
        for x in 0..IMG_SIZE {
            let val = pattern[y * IMG_SIZE + x].clamp(0.0, 1.0);
            let char_idx = (val * 4.0).round() as usize;
            let char_idx = char_idx.min(4);
            print!("{}", chars[char_idx]);
        }
        println!();
    }
}

// =============================================================================
// Target Pattern Generators (Rust versions for display)
// =============================================================================

fn generate_circle_pattern() -> Vec<f32> {
    let mut pattern = vec![0.0; IMG_DIM];
    let center = IMG_SIZE as f32 / 2.0;
    let radius = IMG_SIZE as f32 / 3.0;

    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let dx = x as f32 + 0.5 - center;
            let dy = y as f32 + 0.5 - center;
            let dist = (dx * dx + dy * dy).sqrt();

            // Soft circle with smooth edge
            let val = 1.0 - (dist - radius).max(0.0) / 2.0;
            pattern[y * IMG_SIZE + x] = val.clamp(0.0, 1.0);
        }
    }
    pattern
}

fn generate_cross_pattern() -> Vec<f32> {
    let mut pattern = vec![0.0; IMG_DIM];
    let center = IMG_SIZE / 2;
    let thickness = 2;

    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let in_horizontal = y >= center - thickness && y < center + thickness;
            let in_vertical = x >= center - thickness && x < center + thickness;

            if in_horizontal || in_vertical {
                pattern[y * IMG_SIZE + x] = 1.0;
            }
        }
    }
    pattern
}

fn generate_checkerboard_pattern() -> Vec<f32> {
    let mut pattern = vec![0.0; IMG_DIM];
    let cell_size = 4;

    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let cell_x = x / cell_size;
            let cell_y = y / cell_size;

            if (cell_x + cell_y) % 2 == 0 {
                pattern[y * IMG_SIZE + x] = 1.0;
            }
        }
    }
    pattern
}

// =============================================================================
// WGSL Loss Function Generators
// =============================================================================

/// Generate WGSL for circle pattern MSE loss
fn generate_circle_wgsl() -> temper::expr::Expr {
    // Use unclamped MSE so gradients work everywhere (not just in [0,1])
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

        // Goal: soft circle (values in [0, 1])
        let goal = clamp(1.0 - max(0.0, dist - radius) / 2.0, 0.0, 1.0);

        // Unclamped MSE - gradient works everywhere
        let diff = pos[i] - goal;
        mse = mse + diff * diff;
    }
    return mse / 256.0;
}
"#;

    let grad_code = r#"
fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    let size = 16.0;
    let center = size / 2.0;
    let radius = size / 3.0;

    let y = f32(d_idx / 16u);
    let x = f32(d_idx % 16u);

    let dx = x + 0.5 - center;
    let dy = y + 0.5 - center;
    let dist = sqrt(dx * dx + dy * dy);

    let goal = clamp(1.0 - max(0.0, dist - radius) / 2.0, 0.0, 1.0);

    // Unclamped gradient - pulls toward goal from anywhere
    // Scale up gradient for faster convergence (256 dims means small per-dim gradient otherwise)
    return 2.0 * (pos[d_idx] - goal);
}
"#;

    custom_wgsl(loss_code, grad_code)
}

/// Generate WGSL for cross pattern MSE loss
fn generate_cross_wgsl() -> temper::expr::Expr {
    let loss_code = r#"
fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    var mse = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let y = i / 16u;
        let x = i % 16u;

        // Cross: horizontal bar at y=6,7,8,9 or vertical bar at x=6,7,8,9
        let in_horiz = y >= 6u && y < 10u;
        let in_vert = x >= 6u && x < 10u;
        var goal = 0.0;
        if in_horiz || in_vert {
            goal = 1.0;
        }

        let diff = pos[i] - goal;
        mse = mse + diff * diff;
    }
    return mse / 256.0;
}
"#;

    let grad_code = r#"
fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    let y = d_idx / 16u;
    let x = d_idx % 16u;

    let in_horiz = y >= 6u && y < 10u;
    let in_vert = x >= 6u && x < 10u;
    var goal = 0.0;
    if in_horiz || in_vert {
        goal = 1.0;
    }

    return 2.0 * (pos[d_idx] - goal);
}
"#;

    custom_wgsl(loss_code, grad_code)
}

/// Generate WGSL for checkerboard pattern MSE loss
fn generate_checkerboard_wgsl() -> temper::expr::Expr {
    let loss_code = r#"
fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    var mse = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let y = i / 16u;
        let x = i % 16u;

        // Checkerboard with 4x4 cells
        let cell_x = x / 4u;
        let cell_y = y / 4u;
        var goal = 0.0;
        if (cell_x + cell_y) % 2u == 0u {
            goal = 1.0;
        }

        let diff = pos[i] - goal;
        mse = mse + diff * diff;
    }
    return mse / 256.0;
}
"#;

    let grad_code = r#"
fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    let y = d_idx / 16u;
    let x = d_idx % 16u;

    let cell_x = x / 4u;
    let cell_y = y / 4u;
    var goal = 0.0;
    if (cell_x + cell_y) % 2u == 0u {
        goal = 1.0;
    }

    return 2.0 * (pos[d_idx] - goal);
}
"#;

    custom_wgsl(loss_code, grad_code)
}

/// Generate WGSL for mixture of patterns (min MSE to any pattern)
fn generate_mixture_wgsl() -> temper::expr::Expr {
    let loss_code = r#"
// Compute MSE to circle pattern (unclamped)
fn mse_circle(pos: array<f32, 256>) -> f32 {
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
        let goal = clamp(1.0 - max(0.0, dist - radius) / 2.0, 0.0, 1.0);

        let diff = pos[i] - goal;
        mse = mse + diff * diff;
    }
    return mse / 256.0;
}

// Compute MSE to cross pattern (unclamped)
fn mse_cross(pos: array<f32, 256>) -> f32 {
    var mse = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let y = i / 16u;
        let x = i % 16u;

        let in_horiz = y >= 6u && y < 10u;
        let in_vert = x >= 6u && x < 10u;
        var goal = 0.0;
        if in_horiz || in_vert {
            goal = 1.0;
        }

        let diff = pos[i] - goal;
        mse = mse + diff * diff;
    }
    return mse / 256.0;
}

fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    // Return minimum MSE (particle can converge to either pattern)
    let e_circle = mse_circle(pos);
    let e_cross = mse_cross(pos);
    return min(e_circle, e_cross);
}
"#;

    let grad_code = r#"
// Gradient helpers
fn circle_goal(i: u32) -> f32 {
    let size = 16.0;
    let center = size / 2.0;
    let radius = size / 3.0;

    let y = f32(i / 16u);
    let x = f32(i % 16u);

    let dx = x + 0.5 - center;
    let dy = y + 0.5 - center;
    let dist = sqrt(dx * dx + dy * dy);
    return clamp(1.0 - max(0.0, dist - radius) / 2.0, 0.0, 1.0);
}

fn cross_goal(i: u32) -> f32 {
    let y = i / 16u;
    let x = i % 16u;

    let in_horiz = y >= 6u && y < 10u;
    let in_vert = x >= 6u && x < 10u;
    if in_horiz || in_vert {
        return 1.0;
    }
    return 0.0;
}

fn mse_circle_val(pos: array<f32, 256>) -> f32 {
    var mse = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let diff = pos[i] - circle_goal(i);
        mse = mse + diff * diff;
    }
    return mse / 256.0;
}

fn mse_cross_val(pos: array<f32, 256>) -> f32 {
    var mse = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let diff = pos[i] - cross_goal(i);
        mse = mse + diff * diff;
    }
    return mse / 256.0;
}

fn custom_gradient(pos: array<f32, 256>, dim: u32, d_idx: u32) -> f32 {
    // Gradient toward whichever pattern is closer
    let e_circle = mse_circle_val(pos);
    let e_cross = mse_cross_val(pos);

    if e_circle < e_cross {
        let goal = circle_goal(d_idx);
        return 2.0 * (pos[d_idx] - goal);
    } else {
        let goal = cross_goal(d_idx);
        return 2.0 * (pos[d_idx] - goal);
    }
}
"#;

    custom_wgsl(loss_code, grad_code)
}
