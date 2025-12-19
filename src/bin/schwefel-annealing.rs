//! Schwefel Annealing Visualization
//!
//! Visualizes simulated annealing on the Schwefel function - highly deceptive
//! with a global minimum at (420.97, 420.97), far from the origin where most
//! optimization starts. Local minima are scattered throughout.
//!
//! Run with: cargo run --release --features "viz gpu" --bin schwefel-annealing

use iced::keyboard;
use iced::mouse;
use iced::widget::{Column, Text, canvas, container};
use iced::{Color, Element, Event, Length, Point, Rectangle, Renderer, Subscription, Theme, event};
use temper::thermodynamic::{LossFunction, ThermodynamicParticle, ThermodynamicSystem};

const PARTICLE_COUNT: usize = 1000; // More particles to explore
const DIM: usize = 2;

// Annealing schedule - Schwefel needs MUCH more exploration
const TOTAL_STEPS: u32 = 15000;
const T_START: f32 = 500.0; // Very hot start to explore 1000x1000 domain
const T_END: f32 = 0.1; // Don't cool too much - need mobility in big space

// Domain (Schwefel's global min is at 420.97, so we need a large domain)
const DOMAIN_MIN: f32 = -500.0;
const DOMAIN_MAX: f32 = 500.0;
const DOMAIN_SIZE: f32 = DOMAIN_MAX - DOMAIN_MIN;

// Global optimum location
const GLOBAL_OPT: f32 = 420.9687;

fn main() -> iced::Result {
    iced::application(App::new, update, view)
        .subscription(subscription)
        .title("Schwefel Annealing - Deceptive Global Minimum")
        .run()
}

// Schwefel function for rendering heatmap
fn schwefel(x: f32, y: f32) -> f32 {
    let n = 2.0;
    418.9829 * n - x * (x.abs().sqrt()).sin() - y * (y.abs().sqrt()).sin()
}

struct App {
    system: ThermodynamicSystem,
    particles: Vec<ThermodynamicParticle>,
    cache: canvas::Cache,
    heatmap_cache: canvas::Cache,
    step_count: u32,
    temperature: f32,
    paused: bool,
    speed: u32,
    min_energy: f32,
    mean_energy: f32,
    best_pos: [f32; 2],
    energy_history: Vec<f32>,
}

impl App {
    fn new() -> Self {
        let mut system = ThermodynamicSystem::with_loss_function(
            PARTICLE_COUNT,
            DIM,
            T_START,
            LossFunction::Schwefel,
        );
        let particles = system.read_particles();

        Self {
            system,
            particles,
            cache: canvas::Cache::new(),
            heatmap_cache: canvas::Cache::new(),
            step_count: 0,
            temperature: T_START,
            paused: false,
            speed: 10, // Faster default for more steps
            min_energy: f32::MAX,
            mean_energy: 0.0,
            best_pos: [0.0, 0.0],
            energy_history: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.system = ThermodynamicSystem::with_loss_function(
            PARTICLE_COUNT,
            DIM,
            T_START,
            LossFunction::Schwefel,
        );
        self.particles = self.system.read_particles();
        self.step_count = 0;
        self.temperature = T_START;
        self.min_energy = f32::MAX;
        self.mean_energy = 0.0;
        self.best_pos = [0.0, 0.0];
        self.energy_history.clear();
        self.cache.clear();
    }

    fn compute_temperature(&self) -> f32 {
        let progress = (self.step_count as f32 / TOTAL_STEPS as f32).min(1.0);
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
            if app.paused || app.step_count >= TOTAL_STEPS {
                return;
            }

            for _ in 0..app.speed {
                if app.step_count >= TOTAL_STEPS {
                    break;
                }

                app.temperature = app.compute_temperature();
                app.system.set_temperature(app.temperature);
                app.system.step();
                app.step_count += 1;
            }

            app.particles = app.system.read_particles();

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

            if app.step_count % 10 == 0 {
                app.energy_history.push(app.min_energy);
            }

            app.cache.clear();
        }
        Message::TogglePause => app.paused = !app.paused,
        Message::Reset => app.reset(),
        Message::SpeedUp => app.speed = (app.speed * 2).min(50),
        Message::SpeedDown => app.speed = (app.speed / 2).max(1),
        Message::Event(Event::Keyboard(keyboard::Event::KeyPressed { key, .. })) => match key {
            keyboard::Key::Named(keyboard::key::Named::Space) => app.paused = !app.paused,
            keyboard::Key::Character(c) => match c.as_str() {
                "r" => app.reset(),
                "+" | "=" => app.speed = (app.speed * 2).min(50),
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

    let info = format!(
        "Step: {}/{} ({:.0}%) | T: {:.2} | Status: {} | Speed: {}x",
        app.step_count, TOTAL_STEPS, progress, app.temperature, status, app.speed
    );

    let dist_to_opt =
        ((app.best_pos[0] - GLOBAL_OPT).powi(2) + (app.best_pos[1] - GLOBAL_OPT).powi(2)).sqrt();
    let stats = format!(
        "Min Loss: {:.2} | Best: ({:.1}, {:.1}) | Global: ({:.1}, {:.1}) | Dist: {:.1}",
        app.min_energy, app.best_pos[0], app.best_pos[1], GLOBAL_OPT, GLOBAL_OPT, dist_to_opt
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
        let heatmap = self.heatmap_cache.draw(renderer, bounds.size(), |frame| {
            let size = bounds.size();
            let resolution = 100;
            let max_loss = 1700.0; // Schwefel max in domain

            for i in 0..resolution {
                for j in 0..resolution {
                    let x = DOMAIN_MIN + (i as f32 / resolution as f32) * DOMAIN_SIZE;
                    let y = DOMAIN_MIN + (j as f32 / resolution as f32) * DOMAIN_SIZE;
                    let loss = schwefel(x, y);

                    let t = (loss / max_loss).clamp(0.0, 1.0);

                    // Schwefel colors: deep blue for global min, reds for high loss
                    // Multiple local minima will show as blue-ish spots
                    let color = if t < 0.02 {
                        // Near global minimum - bright green
                        Color::from_rgb(0.0, 1.0 - t * 20.0, 0.2)
                    } else if t < 0.2 {
                        // Local minima regions - blue shades
                        Color::from_rgb(0.0, 0.2, 0.5 + t * 2.0)
                    } else if t < 0.5 {
                        // Medium loss - purple
                        Color::from_rgb(0.3 + t * 0.4, 0.1, 0.5 - t * 0.3)
                    } else {
                        // High loss - dark red
                        Color::from_rgb(0.4 + t * 0.3, 0.05, 0.1)
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

            // Mark global optimum at (420.97, 420.97)
            let ox = ((GLOBAL_OPT - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
            let oy = ((GLOBAL_OPT - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;
            let target = canvas::Path::circle(Point::new(ox, oy), 12.0);
            frame.stroke(
                &target,
                canvas::Stroke::default()
                    .with_color(Color::from_rgb(0.0, 1.0, 0.0))
                    .with_width(3.0),
            );

            // Mark origin for reference
            let origin_x = ((0.0 - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
            let origin_y = ((0.0 - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;
            let origin_marker = canvas::Path::circle(Point::new(origin_x, origin_y), 6.0);
            frame.stroke(
                &origin_marker,
                canvas::Stroke::default()
                    .with_color(Color::from_rgb(1.0, 1.0, 1.0))
                    .with_width(2.0),
            );
        });

        let particles_geo = self.cache.draw(renderer, bounds.size(), |frame| {
            let size = bounds.size();

            for p in &self.particles {
                let px = p.pos[0].to_f32();
                let py = p.pos[1].to_f32();
                if px.is_nan() || py.is_nan() {
                    continue;
                }
                let x = ((px - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
                let y = ((py - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;

                let t = (p.energy / 800.0).clamp(0.0, 1.0);
                let color = Color::from_rgb(1.0, 1.0 - t * 0.6, 1.0 - t);

                let circle = canvas::Path::circle(Point::new(x, y), 3.0);
                frame.fill(&circle, color);
            }

            // Best particle
            let bx = ((self.best_pos[0] - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
            let by = ((self.best_pos[1] - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;
            let best = canvas::Path::circle(Point::new(bx, by), 8.0);
            frame.stroke(
                &best,
                canvas::Stroke::default()
                    .with_color(Color::from_rgb(1.0, 1.0, 0.0))
                    .with_width(2.0),
            );

            // Temperature bar
            let bar_width = 20.0;
            let bar_height = size.height - 60.0;
            let bar_x = 15.0;
            let bar_y = 30.0;

            let bg = canvas::Path::rectangle(
                Point::new(bar_x, bar_y),
                iced::Size::new(bar_width, bar_height),
            );
            frame.fill(&bg, Color::from_rgba(0.0, 0.0, 0.0, 0.5));

            let log_t = (self.temperature.ln() - T_END.ln()) / (T_START.ln() - T_END.ln());
            let fill_height = (log_t.clamp(0.0, 1.0) * bar_height) as f32;
            let temp_fill = canvas::Path::rectangle(
                Point::new(bar_x, bar_y + bar_height - fill_height),
                iced::Size::new(bar_width, fill_height),
            );
            let temp_color =
                Color::from_rgb(log_t.clamp(0.0, 1.0), 0.2, 1.0 - log_t.clamp(0.0, 1.0));
            frame.fill(&temp_fill, temp_color);

            // Energy history
            if !self.energy_history.is_empty() {
                let graph_width = 200.0;
                let graph_height = 80.0;
                let graph_x = size.width - graph_width - 20.0;
                let graph_y = size.height - graph_height - 20.0;

                let graph_bg = canvas::Path::rectangle(
                    Point::new(graph_x, graph_y),
                    iced::Size::new(graph_width, graph_height),
                );
                frame.fill(&graph_bg, Color::from_rgba(0.0, 0.0, 0.0, 0.7));

                let len = self.energy_history.len();
                let max_e = self.energy_history.iter().cloned().fold(1.0_f32, f32::max);
                let min_e = 0.1_f32;

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
