//! RGB Image Diffusion Visualization
//!
//! Interactive visualization of thermodynamic color image generation.
//! Watch 16x16 RGB images emerge from noise through temperature annealing.
//!
//! Controls:
//!   Space = Pause/Resume
//!   R = Reset
//!   S = Switch pattern (RedCircle → Rainbow → Checkerboard → Rings → ...)
//!   +/- = Speed up/down
//!
//! Run with: cargo run --release --features "viz gpu" --bin rgb-image-diffusion-viz

use iced::keyboard;
use iced::mouse;
use iced::widget::{Column, Row, Text, canvas, container};
use iced::{Color, Element, Event, Length, Point, Rectangle, Renderer, Subscription, Theme, event};
use temper::ThermodynamicSystem;
use temper::expr::custom_wgsl;

const IMG_SIZE: usize = 16;
const CHANNEL_DIM: usize = IMG_SIZE * IMG_SIZE; // 256 per channel
const RGB_DIM: usize = CHANNEL_DIM * 3; // 768 total
const PARTICLE_COUNT: usize = 50;

// Annealing schedule
const TOTAL_STEPS: u32 = 5000;
const T_START: f32 = 1.0;
const T_END: f32 = 0.001;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Pattern {
    RedCircle,
    Rainbow,
    Checkerboard,
    Rings,
}

impl Pattern {
    fn next(self) -> Self {
        match self {
            Pattern::RedCircle => Pattern::Rainbow,
            Pattern::Rainbow => Pattern::Checkerboard,
            Pattern::Checkerboard => Pattern::Rings,
            Pattern::Rings => Pattern::RedCircle,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Pattern::RedCircle => "Red Circle",
            Pattern::Rainbow => "Rainbow",
            Pattern::Checkerboard => "Red/Green Check",
            Pattern::Rings => "Color Rings",
        }
    }
}

fn main() -> iced::Result {
    iced::application(App::new, update, view)
        .subscription(subscription)
        .title("RGB Image Diffusion - Color Image Generation")
        .run()
}

struct App {
    system: ThermodynamicSystem,
    current_r: Vec<f32>,
    current_g: Vec<f32>,
    current_b: Vec<f32>,
    target_r: Vec<f32>,
    target_g: Vec<f32>,
    target_b: Vec<f32>,
    pattern: Pattern,
    cache: canvas::Cache,
    step_count: u32,
    temperature: f32,
    paused: bool,
    speed: u32,
    min_energy: f32,
    mean_energy: f32,
}

impl App {
    fn new() -> Self {
        let pattern = Pattern::RedCircle;
        let system = create_system(pattern);
        let (target_r, target_g, target_b) = generate_target(pattern);

        Self {
            system,
            current_r: vec![0.5; CHANNEL_DIM],
            current_g: vec![0.5; CHANNEL_DIM],
            current_b: vec![0.5; CHANNEL_DIM],
            target_r,
            target_g,
            target_b,
            pattern,
            cache: canvas::Cache::new(),
            step_count: 0,
            temperature: T_START,
            paused: false,
            speed: 10,
            min_energy: f32::MAX,
            mean_energy: 0.0,
        }
    }

    fn reset(&mut self) {
        self.system = create_system(self.pattern);
        let (r, g, b) = generate_target(self.pattern);
        self.target_r = r;
        self.target_g = g;
        self.target_b = b;
        self.current_r = vec![0.5; CHANNEL_DIM];
        self.current_g = vec![0.5; CHANNEL_DIM];
        self.current_b = vec![0.5; CHANNEL_DIM];
        self.step_count = 0;
        self.temperature = T_START;
        self.min_energy = f32::MAX;
        self.mean_energy = 0.0;
        self.cache.clear();
    }

    fn switch_pattern(&mut self) {
        self.pattern = self.pattern.next();
        self.reset();
    }

    fn compute_temperature(&self) -> f32 {
        let progress = (self.step_count as f32 / TOTAL_STEPS as f32).min(1.0);
        T_START * (T_END / T_START).powf(progress)
    }
}

fn create_system(pattern: Pattern) -> ThermodynamicSystem {
    let expr = match pattern {
        Pattern::RedCircle => generate_red_circle_wgsl(),
        Pattern::Rainbow => generate_rainbow_wgsl(),
        Pattern::Checkerboard => generate_checkerboard_wgsl(),
        Pattern::Rings => generate_rings_wgsl(),
    };
    ThermodynamicSystem::with_expr(PARTICLE_COUNT, RGB_DIM, T_START, expr)
}

fn generate_target(pattern: Pattern) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    match pattern {
        Pattern::RedCircle => generate_red_circle_target(),
        Pattern::Rainbow => generate_rainbow_target(),
        Pattern::Checkerboard => generate_checkerboard_target(),
        Pattern::Rings => generate_rings_target(),
    }
}

#[derive(Debug, Clone)]
enum Message {
    Tick,
    Event(Event),
}

fn update(app: &mut App, message: Message) {
    match message {
        Message::Tick => {
            if !app.paused && app.step_count < TOTAL_STEPS {
                for _ in 0..app.speed {
                    if app.step_count < TOTAL_STEPS {
                        app.temperature = app.compute_temperature();
                        app.system.set_temperature(app.temperature);
                        app.system.step();
                        app.step_count += 1;
                    }
                }

                // Find best particle and extract RGB
                let particles = app.system.read_particles();
                let mut best_energy = f32::MAX;
                let mut best_idx = 0;
                let mut sum_energy = 0.0;
                let mut count = 0;

                for (i, p) in particles.iter().enumerate() {
                    if p.energy.is_finite() {
                        sum_energy += p.energy;
                        count += 1;
                        if p.energy < best_energy {
                            best_energy = p.energy;
                            best_idx = i;
                        }
                    }
                }

                app.min_energy = best_energy;
                app.mean_energy = if count > 0 {
                    sum_energy / count as f32
                } else {
                    0.0
                };

                // Extract best particle's RGB image
                if count > 0 {
                    for i in 0..CHANNEL_DIM {
                        app.current_r[i] = particles[best_idx].pos[i].to_f32().clamp(0.0, 1.0);
                        app.current_g[i] = particles[best_idx].pos[CHANNEL_DIM + i]
                            .to_f32()
                            .clamp(0.0, 1.0);
                        app.current_b[i] = particles[best_idx].pos[2 * CHANNEL_DIM + i]
                            .to_f32()
                            .clamp(0.0, 1.0);
                    }
                }

                app.cache.clear();
            }
        }
        Message::Event(event) => {
            if let Event::Keyboard(keyboard::Event::KeyPressed { key, .. }) = event {
                match key.as_ref() {
                    keyboard::Key::Named(keyboard::key::Named::Space) => {
                        app.paused = !app.paused;
                    }
                    keyboard::Key::Character("r") => {
                        app.reset();
                    }
                    keyboard::Key::Character("s") => {
                        app.switch_pattern();
                    }
                    keyboard::Key::Character("+") | keyboard::Key::Character("=") => {
                        app.speed = (app.speed * 2).min(100);
                    }
                    keyboard::Key::Character("-") => {
                        app.speed = (app.speed / 2).max(1);
                    }
                    _ => {}
                }
            }
        }
    }
}

fn view(app: &App) -> Element<'_, Message> {
    let progress = (app.step_count as f32 / TOTAL_STEPS as f32) * 100.0;
    let status = if app.paused {
        "PAUSED"
    } else if app.step_count >= TOTAL_STEPS {
        "DONE"
    } else {
        "RUNNING"
    };

    let title = format!(
        "Pattern: {} | Step: {}/{} ({:.0}%) | {}",
        app.pattern.name(),
        app.step_count,
        TOTAL_STEPS,
        progress,
        status
    );

    let stats = format!(
        "Temperature: {:.4} | Energy: {:.4} (mean: {:.3}) | Speed: {}x | Dims: 768 (RGB)",
        app.temperature, app.min_energy, app.mean_energy, app.speed
    );

    let controls = "Space=Pause  R=Reset  S=Switch Pattern  +/-=Speed";

    Column::new()
        .push(Text::new(title).size(18))
        .push(Text::new(stats).size(14))
        .push(Text::new(controls).size(12))
        .push(
            Row::new()
                .push(
                    Column::new()
                        .push(Text::new("Target").size(14))
                        .push(container(
                            canvas(TargetCanvas {
                                r: &app.target_r,
                                g: &app.target_g,
                                b: &app.target_b,
                            })
                            .width(Length::Fixed(320.0))
                            .height(Length::Fixed(320.0)),
                        )),
                )
                .push(
                    Column::new()
                        .push(Text::new("Generated").size(14))
                        .push(container(
                            canvas(GeneratedCanvas {
                                r: &app.current_r,
                                g: &app.current_g,
                                b: &app.current_b,
                                cache: &app.cache,
                            })
                            .width(Length::Fixed(320.0))
                            .height(Length::Fixed(320.0)),
                        )),
                )
                .spacing(20),
        )
        .spacing(5)
        .padding(10)
        .into()
}

fn subscription(_app: &App) -> Subscription<Message> {
    Subscription::batch([
        iced::time::every(std::time::Duration::from_millis(16)).map(|_| Message::Tick),
        event::listen().map(Message::Event),
    ])
}

// =============================================================================
// Canvas Rendering
// =============================================================================

struct TargetCanvas<'a> {
    r: &'a [f32],
    g: &'a [f32],
    b: &'a [f32],
}

impl canvas::Program<Message> for TargetCanvas<'_> {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        draw_rgb_image(&mut frame, self.r, self.g, self.b, bounds.size());
        vec![frame.into_geometry()]
    }
}

struct GeneratedCanvas<'a> {
    r: &'a [f32],
    g: &'a [f32],
    b: &'a [f32],
    cache: &'a canvas::Cache,
}

impl canvas::Program<Message> for GeneratedCanvas<'_> {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
            draw_rgb_image(frame, self.r, self.g, self.b, bounds.size());
        });
        vec![geometry]
    }
}

fn draw_rgb_image(frame: &mut canvas::Frame, r: &[f32], g: &[f32], b: &[f32], size: iced::Size) {
    let cell_w = size.width / IMG_SIZE as f32;
    let cell_h = size.height / IMG_SIZE as f32;

    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let idx = y * IMG_SIZE + x;
            let color = Color::from_rgb(r[idx], g[idx], b[idx]);
            let rect = canvas::Path::rectangle(
                Point::new(x as f32 * cell_w, y as f32 * cell_h),
                iced::Size::new(cell_w, cell_h),
            );
            frame.fill(&rect, color);
        }
    }

    // Draw grid lines
    let grid_color = Color::from_rgba(0.3, 0.3, 0.3, 0.3);
    for i in 0..=IMG_SIZE {
        let x = i as f32 * cell_w;
        let y = i as f32 * cell_h;

        let vline = canvas::Path::line(Point::new(x, 0.0), Point::new(x, size.height));
        frame.stroke(
            &vline,
            canvas::Stroke::default()
                .with_color(grid_color)
                .with_width(0.5),
        );

        let hline = canvas::Path::line(Point::new(0.0, y), Point::new(size.width, y));
        frame.stroke(
            &hline,
            canvas::Stroke::default()
                .with_color(grid_color)
                .with_width(0.5),
        );
    }
}

// =============================================================================
// Pattern Generators (Rust)
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
                r[idx] = 1.0;
                g[idx] = 0.0;
                b[idx] = 0.0;
            } else {
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

fn generate_checkerboard_target() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = vec![0.0; CHANNEL_DIM];
    let mut g = vec![0.0; CHANNEL_DIM];
    let mut b = vec![0.0; CHANNEL_DIM];

    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let idx = y * IMG_SIZE + x;
            let cell_x = x / 4;
            let cell_y = y / 4;
            if (cell_x + cell_y) % 2 == 0 {
                r[idx] = 1.0;
                g[idx] = 0.0;
                b[idx] = 0.0;
            } else {
                r[idx] = 0.0;
                g[idx] = 1.0;
                b[idx] = 0.0;
            }
        }
    }
    (r, g, b)
}

fn generate_rings_target() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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

            let hue = (dist / (IMG_SIZE as f32 / 2.0)).fract();
            let (ri, gi, bi) = hsv_to_rgb(hue, 1.0, 1.0);
            r[idx] = ri;
            g[idx] = gi;
            b[idx] = bi;
        }
    }
    (r, g, b)
}

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

        var goal_r = 0.0;
        var goal_g = 0.0;
        var goal_b = 0.8;
        if dist < radius {
            goal_r = 1.0;
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

    let size = 16.0;
    let center = size / 2.0;
    let radius = size / 3.0;

    let channel = d_idx / 256u;
    let pixel_idx = d_idx % 256u;

    let y = f32(pixel_idx / 16u);
    let x = f32(pixel_idx % 16u);

    let dx = x + 0.5 - center;
    let dy = y + 0.5 - center;
    let dist = sqrt(dx * dx + dy * dy);

    var goal = 0.0;
    if channel == 0u {
        if dist < radius { goal = 1.0; }
    } else if channel == 1u {
        goal = 0.0;
    } else {
        if dist >= radius { goal = 0.8; }
    }

    return 2.0 * (pos[d_idx] - goal);
}
"#;

    custom_wgsl(loss_code, grad_code)
}

fn generate_rainbow_wgsl() -> temper::expr::Expr {
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

fn generate_checkerboard_wgsl() -> temper::expr::Expr {
    let loss_code = r#"
fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    var mse = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let y = i / 16u;
        let x = i % 16u;

        let cell_x = x / 4u;
        let cell_y = y / 4u;

        var goal_r = 0.0;
        var goal_g = 1.0;
        var goal_b = 0.0;
        if (cell_x + cell_y) % 2u == 0u {
            goal_r = 1.0;
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
        if is_red { goal = 1.0; }
    } else if channel == 1u {
        if !is_red { goal = 1.0; }
    }

    return 2.0 * (pos[d_idx] - goal);
}
"#;

    custom_wgsl(loss_code, grad_code)
}

fn generate_rings_wgsl() -> temper::expr::Expr {
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
