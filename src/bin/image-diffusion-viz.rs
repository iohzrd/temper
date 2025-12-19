//! Image Diffusion Visualization
//!
//! Interactive visualization of thermodynamic image generation.
//! Watch 16x16 images emerge from noise through temperature annealing.
//!
//! Controls:
//!   Space = Pause/Resume
//!   R = Reset
//!   S = Switch pattern (Circle → Cross → Checkerboard → Circle...)
//!   +/- = Speed up/down
//!
//! Run with: cargo run --release --features "viz gpu" --bin image-diffusion-viz

use iced::keyboard;
use iced::mouse;
use iced::widget::{Column, Row, Text, canvas, container};
use iced::{Color, Element, Event, Length, Point, Rectangle, Renderer, Subscription, Theme, event};
use temper::ThermodynamicSystem;
use temper::expr::custom_wgsl;

const IMG_SIZE: usize = 16;
const IMG_DIM: usize = IMG_SIZE * IMG_SIZE;
const PARTICLE_COUNT: usize = 500; // More particles for better exploration

// Optimization schedule - using leapfrog only (no accept/reject)
const TOTAL_STEPS: u32 = 2000;
const T_START: f32 = 0.001; // Very low T = near-zero momentum = gradient descent
const T_END: f32 = 0.0001; // Stay low

#[derive(Debug, Clone, Copy, PartialEq)]
enum Pattern {
    Circle,
    Cross,
    Checkerboard,
}

impl Pattern {
    fn next(self) -> Self {
        match self {
            Pattern::Circle => Pattern::Cross,
            Pattern::Cross => Pattern::Checkerboard,
            Pattern::Checkerboard => Pattern::Circle,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Pattern::Circle => "Circle",
            Pattern::Cross => "Cross",
            Pattern::Checkerboard => "Checkerboard",
        }
    }
}

fn main() -> iced::Result {
    iced::application(App::new, update, view)
        .subscription(subscription)
        .title("Image Diffusion - Thermodynamic Image Generation")
        .run()
}

struct App {
    system: ThermodynamicSystem,
    current_image: Vec<f32>,
    target_image: Vec<f32>,
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
        let pattern = Pattern::Circle;
        let system = create_system(pattern);
        let target_image = generate_target(pattern);
        let current_image = vec![0.5; IMG_DIM]; // Start with gray

        Self {
            system,
            current_image,
            target_image,
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
        self.target_image = generate_target(self.pattern);
        self.current_image = vec![0.5; IMG_DIM];
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
        Pattern::Circle => generate_circle_wgsl(),
        Pattern::Cross => generate_cross_wgsl(),
        Pattern::Checkerboard => generate_checkerboard_wgsl(),
    };
    let mut system = ThermodynamicSystem::with_expr(PARTICLE_COUNT, IMG_DIM, T_START, expr);

    // Leapfrog parameters - slower for visible convergence
    // Effective learning rate = ε²/(2m), so ε=0.05, m=1 → LR=0.00125
    system.set_leapfrog_steps(1); // Single step per iteration
    system.set_step_size(0.05); // Small step for visible convergence
    system.set_mass(1.0); // Standard mass

    system
}

fn generate_target(pattern: Pattern) -> Vec<f32> {
    match pattern {
        Pattern::Circle => generate_circle_pattern(),
        Pattern::Cross => generate_cross_pattern(),
        Pattern::Checkerboard => generate_checkerboard_pattern(),
    }
}

#[derive(Debug, Clone)]
enum Message {
    Tick,
    TogglePause,
    Reset,
    SwitchPattern,
    SpeedUp,
    SpeedDown,
    Event(Event),
}

fn update(app: &mut App, message: Message) {
    match message {
        Message::Tick => {
            if app.paused {
                return;
            }

            for _ in 0..app.speed {
                if app.step_count >= TOTAL_STEPS {
                    break;
                }

                app.temperature = app.compute_temperature();
                app.system.set_temperature(app.temperature);
                app.system.step_leapfrog_only(); // Pure gradient descent, no accept/reject
                app.step_count += 1;
            }

            // Read best particle
            let particles = app.system.read_particles();
            let mut sum_energy = 0.0;
            let mut count = 0;
            let mut best_idx = 0;
            let mut best_energy = f32::MAX;

            for (i, p) in particles.iter().enumerate() {
                if !p.energy.is_nan() {
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

            // Extract best particle's image
            if count > 0 {
                for i in 0..IMG_DIM {
                    app.current_image[i] = particles[best_idx].pos[i].to_f32().clamp(0.0, 1.0);
                }
            }

            app.cache.clear();
        }
        Message::TogglePause => {
            app.paused = !app.paused;
        }
        Message::Reset => {
            app.reset();
        }
        Message::SwitchPattern => {
            app.switch_pattern();
        }
        Message::SpeedUp => {
            app.speed = (app.speed * 2).min(100);
        }
        Message::SpeedDown => {
            app.speed = (app.speed / 2).max(1);
        }
        Message::Event(Event::Keyboard(keyboard::Event::KeyPressed { key, .. })) => match key {
            keyboard::Key::Named(keyboard::key::Named::Space) => {
                app.paused = !app.paused;
            }
            keyboard::Key::Character(c) => match c.as_str() {
                "r" => app.reset(),
                "s" => app.switch_pattern(),
                "+" | "=" => app.speed = (app.speed * 2).min(100),
                "-" => app.speed = (app.speed / 2).max(1),
                _ => {}
            },
            _ => {}
        },
        Message::Event(_) => {}
    }
}

fn view(app: &App) -> Element<'_, Message> {
    let progress = (app.step_count as f32 / TOTAL_STEPS as f32 * 100.0).min(100.0);
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
        "Temperature: {:.4} | Energy: {:.4} (mean: {:.3}) | Speed: {}x",
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
                                image: &app.target_image,
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
                                image: &app.current_image,
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

// Canvas for target image (static)
struct TargetCanvas<'a> {
    image: &'a [f32],
}

impl<'a> canvas::Program<Message> for TargetCanvas<'a> {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let geom = canvas::Cache::new().draw(renderer, bounds.size(), |frame| {
            draw_image(frame, self.image, bounds.size());
        });
        vec![geom]
    }
}

// Canvas for generated image (updates each frame)
struct GeneratedCanvas<'a> {
    image: &'a [f32],
    cache: &'a canvas::Cache,
}

impl<'a> canvas::Program<Message> for GeneratedCanvas<'a> {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let geom = self.cache.draw(renderer, bounds.size(), |frame| {
            draw_image(frame, self.image, bounds.size());
        });
        vec![geom]
    }
}

fn draw_image(frame: &mut canvas::Frame, image: &[f32], size: iced::Size) {
    let cell_w = size.width / IMG_SIZE as f32;
    let cell_h = size.height / IMG_SIZE as f32;

    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let idx = y * IMG_SIZE + x;
            let val = image[idx].clamp(0.0, 1.0);

            // Grayscale color
            let color = Color::from_rgb(val, val, val);

            let px = x as f32 * cell_w;
            let py = y as f32 * cell_h;

            let rect = canvas::Path::rectangle(
                Point::new(px, py),
                iced::Size::new(cell_w + 0.5, cell_h + 0.5),
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

fn generate_circle_pattern() -> Vec<f32> {
    let mut pattern = vec![0.0; IMG_DIM];
    let center = IMG_SIZE as f32 / 2.0;
    let radius = IMG_SIZE as f32 / 3.0;

    for y in 0..IMG_SIZE {
        for x in 0..IMG_SIZE {
            let dx = x as f32 + 0.5 - center;
            let dy = y as f32 + 0.5 - center;
            let dist = (dx * dx + dy * dy).sqrt();
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

fn generate_circle_wgsl() -> temper::expr::Expr {
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
        let goal = clamp(1.0 - max(0.0, dist - radius) / 2.0, 0.0, 1.0);

        let diff = pos[i] - goal;
        mse = mse + diff * diff;
    }
    return mse / 256.0;  // Normalize to average MSE
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
    // Gradient of MSE/256 = 2*(p-goal)/256, but scale up for faster learning
    return 2.0 * (pos[d_idx] - goal);
}
"#;

    custom_wgsl(loss_code, grad_code)
}

fn generate_cross_wgsl() -> temper::expr::Expr {
    let loss_code = r#"
fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
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

fn generate_checkerboard_wgsl() -> temper::expr::Expr {
    let loss_code = r#"
fn custom_loss(pos: array<f32, 256>, dim: u32) -> f32 {
    var mse = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let y = i / 16u;
        let x = i % 16u;

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
