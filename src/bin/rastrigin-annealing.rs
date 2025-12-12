//! Rastrigin Annealing Visualization
//!
//! Visualizes simulated annealing on the Rastrigin function - a highly multimodal
//! landscape with many local minima. Watch particles escape local traps via thermal
//! noise and converge to the global minimum at the origin.
//!
//! The background shows the loss surface as a heatmap.
//!
//! Run with: cargo run --release --features "viz gpu" --bin rastrigin-annealing

use iced::keyboard;
use iced::mouse;
use iced::widget::{Column, Text, canvas, container};
use iced::{Color, Element, Event, Length, Point, Rectangle, Renderer, Subscription, Theme, event};
use temper::thermodynamic::{LossFunction, ThermodynamicParticle, ThermodynamicSystem};

const PARTICLE_COUNT: usize = 500;
const DIM: usize = 2;

// Annealing schedule
const TOTAL_STEPS: u32 = 3000;
const T_START: f32 = 5.0;
const T_END: f32 = 0.001;

// Domain (Rastrigin is typically evaluated in [-5.12, 5.12])
const DOMAIN_MIN: f32 = -4.0;
const DOMAIN_MAX: f32 = 4.0;
const DOMAIN_SIZE: f32 = DOMAIN_MAX - DOMAIN_MIN;

fn main() -> iced::Result {
    iced::application(App::new, update, view)
        .subscription(subscription)
        .title("Rastrigin Annealing - Escaping Local Minima")
        .run()
}

// Rastrigin function for rendering heatmap
fn rastrigin(x: f32, y: f32) -> f32 {
    let n = 2.0;
    let a = 10.0;
    a * n
        + (x * x - a * (2.0 * std::f32::consts::PI * x).cos())
        + (y * y - a * (2.0 * std::f32::consts::PI * y).cos())
}

struct App {
    system: ThermodynamicSystem,
    particles: Vec<ThermodynamicParticle>,
    cache: canvas::Cache,
    heatmap_cache: canvas::Cache,
    step_count: u32,
    temperature: f32,
    paused: bool,
    speed: u32, // Steps per frame
    // Statistics
    min_energy: f32,
    mean_energy: f32,
    best_pos: [f32; 2],
    energy_history: Vec<f32>,
    temp_history: Vec<f32>,
}

impl App {
    fn new() -> Self {
        let mut system = ThermodynamicSystem::with_loss_function(
            PARTICLE_COUNT,
            DIM,
            T_START,
            LossFunction::Rastrigin,
        );
        system.set_repulsion_samples(32); // Some repulsion for diversity
        let particles = system.read_particles();

        Self {
            system,
            particles,
            cache: canvas::Cache::new(),
            heatmap_cache: canvas::Cache::new(),
            step_count: 0,
            temperature: T_START,
            paused: false,
            speed: 5,
            min_energy: f32::MAX,
            mean_energy: 0.0,
            best_pos: [0.0, 0.0],
            energy_history: Vec::new(),
            temp_history: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.system = ThermodynamicSystem::with_loss_function(
            PARTICLE_COUNT,
            DIM,
            T_START,
            LossFunction::Rastrigin,
        );
        self.system.set_repulsion_samples(32);
        self.particles = self.system.read_particles();
        self.step_count = 0;
        self.temperature = T_START;
        self.min_energy = f32::MAX;
        self.mean_energy = 0.0;
        self.best_pos = [0.0, 0.0];
        self.energy_history.clear();
        self.temp_history.clear();
        self.cache.clear();
    }

    fn compute_temperature(&self) -> f32 {
        let progress = (self.step_count as f32 / TOTAL_STEPS as f32).min(1.0);
        // Exponential cooling schedule
        T_START * (T_END / T_START).powf(progress)
    }
}

#[derive(Debug, Clone)]
enum Message {
    Tick,
    TogglePause,
    Reset,
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

                // Update temperature
                app.temperature = app.compute_temperature();
                app.system.set_temperature(app.temperature);

                // Reduce repulsion as we cool down
                let repulsion = if app.temperature > 1.0 {
                    32
                } else if app.temperature > 0.1 {
                    16
                } else {
                    0
                };
                app.system.set_repulsion_samples(repulsion);

                // Use smaller dt at low temperatures to prevent overshooting
                // Rastrigin has sharp gradients from cosine terms
                let dt = if app.temperature > 0.1 {
                    0.01
                } else if app.temperature > 0.01 {
                    0.005
                } else {
                    0.002
                };
                app.system.set_dt(dt);

                // Run simulation step
                app.system.step();
                app.step_count += 1;
            }

            // Read particles
            app.particles = app.system.read_particles();

            // Compute statistics
            let mut sum_energy = 0.0;
            let mut count = 0;
            for p in &app.particles {
                if !p.energy.is_nan() {
                    sum_energy += p.energy;
                    count += 1;
                    if p.energy < app.min_energy {
                        app.min_energy = p.energy;
                        app.best_pos = [p.pos[0].to_f32(), p.pos[1].to_f32()];
                    }
                }
            }
            app.mean_energy = if count > 0 {
                sum_energy / count as f32
            } else {
                0.0
            };

            // Record history
            if app.step_count % 10 == 0 {
                app.energy_history.push(app.min_energy);
                app.temp_history.push(app.temperature);
            }

            app.cache.clear();
        }
        Message::TogglePause => {
            app.paused = !app.paused;
        }
        Message::Reset => {
            app.reset();
        }
        Message::SpeedUp => {
            app.speed = (app.speed * 2).min(50);
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
                "+" | "=" => {
                    app.speed = (app.speed * 2).min(50);
                }
                "-" => {
                    app.speed = (app.speed / 2).max(1);
                }
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

    let info = format!(
        "Step: {}/{} ({:.0}%) | T: {:.4} | Status: {} | Speed: {}x",
        app.step_count, TOTAL_STEPS, progress, app.temperature, status, app.speed
    );

    let stats = format!(
        "Min Loss: {:.4} | Mean: {:.2} | Best: ({:.3}, {:.3}) | Global optimum: (0, 0)",
        app.min_energy, app.mean_energy, app.best_pos[0], app.best_pos[1]
    );

    let controls = "Space=Pause  R=Reset  +/-=Speed";

    Column::new()
        .push(Text::new(info).size(16))
        .push(Text::new(stats).size(14))
        .push(Text::new(controls).size(12))
        .push(
            container(canvas(app).width(Length::Fill).height(Length::Fill))
                .width(Length::Fill)
                .height(Length::Fill),
        )
        .into()
}

fn subscription(_app: &App) -> Subscription<Message> {
    Subscription::batch([
        iced::time::every(std::time::Duration::from_millis(16)).map(|_| Message::Tick),
        event::listen().map(Message::Event),
    ])
}

impl canvas::Program<Message> for App {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        // Draw heatmap (cached)
        let heatmap = self.heatmap_cache.draw(renderer, bounds.size(), |frame| {
            let size = bounds.size();
            let resolution = 80; // Grid resolution for heatmap

            // Find max loss for normalization
            let max_loss = 80.0; // Rastrigin max in our domain

            for i in 0..resolution {
                for j in 0..resolution {
                    let x = DOMAIN_MIN + (i as f32 / resolution as f32) * DOMAIN_SIZE;
                    let y = DOMAIN_MIN + (j as f32 / resolution as f32) * DOMAIN_SIZE;
                    let loss = rastrigin(x, y);

                    // Map loss to color (dark blue = low, yellow = high)
                    let t = (loss / max_loss).clamp(0.0, 1.0);
                    let color = if t < 0.1 {
                        // Near global minimum - bright green
                        Color::from_rgb(0.0, 0.4 + t * 2.0, 0.0)
                    } else if t < 0.3 {
                        // Local minima - blue
                        Color::from_rgb(0.0, 0.1, 0.2 + t)
                    } else {
                        // High loss - dark purple/red
                        Color::from_rgb(t * 0.3, 0.0, 0.15)
                    };

                    let px = (i as f32 / resolution as f32) * size.width;
                    let py = (j as f32 / resolution as f32) * size.height;
                    let cell_w = size.width / resolution as f32 + 1.0;
                    let cell_h = size.height / resolution as f32 + 1.0;

                    let rect = canvas::Path::rectangle(
                        Point::new(px, py),
                        iced::Size::new(cell_w, cell_h),
                    );
                    frame.fill(&rect, color);
                }
            }

            // Draw grid lines at integer coordinates
            let grid_color = Color::from_rgba(1.0, 1.0, 1.0, 0.1);
            for i in -4..=4 {
                let x = ((i as f32 - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
                let y = ((i as f32 - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;

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

            // Mark global optimum at origin
            let ox = ((0.0 - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
            let oy = ((0.0 - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;
            let target = canvas::Path::circle(Point::new(ox, oy), 12.0);
            frame.stroke(
                &target,
                canvas::Stroke::default()
                    .with_color(Color::from_rgb(0.0, 1.0, 0.0))
                    .with_width(3.0),
            );
        });

        // Draw particles and UI (not cached - changes every frame)
        let particles_geo = self.cache.draw(renderer, bounds.size(), |frame| {
            let size = bounds.size();

            // Draw particles with temperature-based coloring
            for p in &self.particles {
                let px = p.pos[0].to_f32();
                let py = p.pos[1].to_f32();
                if px.is_nan() || py.is_nan() {
                    continue;
                }
                let x = ((px - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
                let y = ((py - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;

                // Color based on energy (white = low, orange = high)
                let t = (p.energy / 40.0).clamp(0.0, 1.0);
                let color = Color::from_rgb(1.0, 1.0 - t * 0.6, 1.0 - t);

                let circle = canvas::Path::circle(Point::new(x, y), 3.0);
                frame.fill(&circle, color);
            }

            // Draw best particle found
            let bx = ((self.best_pos[0] - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
            let by = ((self.best_pos[1] - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;
            let best = canvas::Path::circle(Point::new(bx, by), 8.0);
            frame.stroke(
                &best,
                canvas::Stroke::default()
                    .with_color(Color::from_rgb(1.0, 1.0, 0.0))
                    .with_width(2.0),
            );

            // Temperature bar (left side)
            let bar_width = 20.0;
            let bar_height = size.height - 60.0;
            let bar_x = 15.0;
            let bar_y = 30.0;

            // Background
            let bg = canvas::Path::rectangle(
                Point::new(bar_x, bar_y),
                iced::Size::new(bar_width, bar_height),
            );
            frame.fill(&bg, Color::from_rgba(0.0, 0.0, 0.0, 0.5));

            // Temperature fill (log scale)
            let log_t = (self.temperature.ln() - T_END.ln()) / (T_START.ln() - T_END.ln());
            let fill_height = (log_t.clamp(0.0, 1.0) * bar_height) as f32;
            let temp_fill = canvas::Path::rectangle(
                Point::new(bar_x, bar_y + bar_height - fill_height),
                iced::Size::new(bar_width, fill_height),
            );
            // Color gradient: red (hot) -> blue (cold)
            let temp_color =
                Color::from_rgb(log_t.clamp(0.0, 1.0), 0.2, 1.0 - log_t.clamp(0.0, 1.0));
            frame.fill(&temp_fill, temp_color);

            // Energy history graph (bottom right)
            if !self.energy_history.is_empty() {
                let graph_width = 200.0;
                let graph_height = 80.0;
                let graph_x = size.width - graph_width - 20.0;
                let graph_y = size.height - graph_height - 20.0;

                // Background
                let graph_bg = canvas::Path::rectangle(
                    Point::new(graph_x, graph_y),
                    iced::Size::new(graph_width, graph_height),
                );
                frame.fill(&graph_bg, Color::from_rgba(0.0, 0.0, 0.0, 0.7));

                // Plot min energy (log scale)
                let len = self.energy_history.len();
                let max_e = self.energy_history.iter().cloned().fold(1.0_f32, f32::max);
                let min_e = 0.001_f32;

                for i in 1..len {
                    let x1 = graph_x + ((i - 1) as f32 / len as f32) * graph_width;
                    let x2 = graph_x + (i as f32 / len as f32) * graph_width;

                    let e1 = self.energy_history[i - 1].max(min_e);
                    let e2 = self.energy_history[i].max(min_e);

                    let y1 = graph_y + graph_height
                        - ((e1.ln() - min_e.ln()) / (max_e.ln() - min_e.ln())).clamp(0.0, 1.0)
                            * graph_height;
                    let y2 = graph_y + graph_height
                        - ((e2.ln() - min_e.ln()) / (max_e.ln() - min_e.ln())).clamp(0.0, 1.0)
                            * graph_height;

                    if !y1.is_nan() && !y2.is_nan() {
                        let line = canvas::Path::line(Point::new(x1, y1), Point::new(x2, y2));
                        frame.stroke(
                            &line,
                            canvas::Stroke::default()
                                .with_color(Color::from_rgb(0.0, 1.0, 1.0))
                                .with_width(2.0),
                        );
                    }
                }
            }
        });

        vec![heatmap, particles_geo]
    }
}
