//! Loss Landscape Visualizer
//!
//! Real-time visualization of particles evolving on 2D energy surfaces.
//! Watch how temperature controls exploration vs exploitation!
//!
//! Controls:
//!   Space = Pause/Resume
//!   R = Reset
//!   L = Switch landscape
//!   T = Toggle temperature mode (annealing vs fixed)
//!   Up/Down = Adjust temperature (in fixed mode)
//!   +/- = Speed up/down
//!
//! Run with: cargo run --release --features "viz gpu" --bin landscape-viz

use iced::keyboard;
use iced::mouse;
use iced::widget::{Column, Text, canvas, container};
use iced::{
    Color, Element, Event, Length, Point, Rectangle, Renderer, Size, Subscription, Theme, event,
};
use temper::ThermodynamicSystem;
use temper::viz::{
    Landscape, X_MAX, X_MIN, Y_MAX, Y_MIN, draw_axes, draw_energy_surface, temperature_to_color,
};

const PARTICLE_COUNT: usize = 100;
const TOTAL_STEPS: u32 = 10000;
const T_START: f32 = 1.0;
const T_END: f32 = 0.001;

struct App {
    system: ThermodynamicSystem,
    landscape: Landscape,
    step_count: u32,
    temperature: f32,
    paused: bool,
    speed: u32,
    annealing: bool,
    fixed_temp: f32,
    particles_x: Vec<f32>,
    particles_y: Vec<f32>,
    min_energy: f32,
    mean_energy: f32,
    cache: canvas::Cache,
    surface_cache: canvas::Cache,
}

impl App {
    fn new() -> (Self, iced::Task<Message>) {
        let landscape = Landscape::Rastrigin;
        let mut system =
            ThermodynamicSystem::with_expr(PARTICLE_COUNT, 2, T_START, landscape.expr());
        // Set bounds to match the visualization range [-5, 5]
        system.set_position_bounds(X_MIN, X_MAX);

        (
            Self {
                system,
                landscape,
                step_count: 0,
                temperature: T_START,
                paused: false,
                speed: 1,
                annealing: true,
                fixed_temp: 0.5,
                particles_x: vec![0.0; PARTICLE_COUNT],
                particles_y: vec![0.0; PARTICLE_COUNT],
                min_energy: 0.0,
                mean_energy: 0.0,
                cache: canvas::Cache::new(),
                surface_cache: canvas::Cache::new(),
            },
            iced::Task::none(),
        )
    }

    fn reset(&mut self) {
        self.system =
            ThermodynamicSystem::with_expr(PARTICLE_COUNT, 2, T_START, self.landscape.expr());
        // Set bounds to match the visualization range [-5, 5]
        self.system.set_position_bounds(X_MIN, X_MAX);
        self.step_count = 0;
        self.temperature = if self.annealing {
            T_START
        } else {
            self.fixed_temp
        };
        self.system.set_temperature(self.temperature);
        self.cache.clear();
    }

    fn switch_landscape(&mut self) {
        self.landscape = self.landscape.next();
        self.surface_cache.clear();
        self.reset();
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
            if !app.paused {
                for _ in 0..app.speed {
                    if app.annealing && app.step_count < TOTAL_STEPS {
                        let progress = app.step_count as f32 / TOTAL_STEPS as f32;
                        app.temperature = T_START * (T_END / T_START).powf(progress);
                    }

                    app.system.set_temperature(app.temperature);
                    app.system.step();

                    if app.annealing {
                        app.step_count = (app.step_count + 1).min(TOTAL_STEPS);
                    }
                }

                // Read particle positions
                let particles = app.system.read_particles();
                let mut sum_energy = 0.0;
                let mut min_energy = f32::MAX;
                let mut count = 0;

                for (i, p) in particles.iter().enumerate() {
                    if p.energy.is_finite() {
                        app.particles_x[i] = p.pos[0].to_f32();
                        app.particles_y[i] = p.pos[1].to_f32();
                        sum_energy += p.energy;
                        if p.energy < min_energy {
                            min_energy = p.energy;
                        }
                        count += 1;
                    }
                }

                app.min_energy = min_energy;
                app.mean_energy = if count > 0 {
                    sum_energy / count as f32
                } else {
                    0.0
                };
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
                    keyboard::Key::Character("l") => {
                        app.switch_landscape();
                    }
                    keyboard::Key::Character("t") => {
                        app.annealing = !app.annealing;
                        if !app.annealing {
                            app.temperature = app.fixed_temp;
                        }
                    }
                    keyboard::Key::Named(keyboard::key::Named::ArrowUp) => {
                        if !app.annealing {
                            app.fixed_temp = (app.fixed_temp * 1.5).min(10.0);
                            app.temperature = app.fixed_temp;
                        }
                    }
                    keyboard::Key::Named(keyboard::key::Named::ArrowDown) => {
                        if !app.annealing {
                            app.fixed_temp = (app.fixed_temp / 1.5).max(0.001);
                            app.temperature = app.fixed_temp;
                        }
                    }
                    keyboard::Key::Character("=") | keyboard::Key::Character("+") => {
                        app.speed = (app.speed * 2).min(64);
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
    let progress = if app.annealing {
        (app.step_count as f32 / TOTAL_STEPS as f32) * 100.0
    } else {
        0.0
    };

    let status = if app.paused {
        "PAUSED"
    } else if app.annealing && app.step_count >= TOTAL_STEPS {
        "DONE"
    } else {
        "RUNNING"
    };

    let mode = if app.annealing {
        format!("Annealing ({:.0}%)", progress)
    } else {
        "Fixed T (Up/Down to adjust)".to_string()
    };

    let title = format!("{} | {} | {}", app.landscape.name(), mode, status);

    let stats = format!(
        "T: {:.4} | Energy: {:.4} (mean: {:.3}) | Speed: {}x",
        app.temperature, app.min_energy, app.mean_energy, app.speed
    );

    let controls = "Space=Pause  R=Reset  L=Landscape  T=Toggle Mode  +/-=Speed";

    Column::new()
        .push(Text::new(title).size(18))
        .push(Text::new(app.landscape.description()).size(12))
        .push(Text::new(stats).size(14))
        .push(Text::new(controls).size(12))
        .push(
            container(
                canvas(LandscapeCanvas {
                    landscape: &app.landscape,
                    particles_x: &app.particles_x,
                    particles_y: &app.particles_y,
                    temperature: app.temperature,
                    cache: &app.cache,
                    surface_cache: &app.surface_cache,
                })
                .width(Length::Fixed(600.0))
                .height(Length::Fixed(600.0)),
            )
            .padding(10),
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

struct LandscapeCanvas<'a> {
    landscape: &'a Landscape,
    particles_x: &'a [f32],
    particles_y: &'a [f32],
    temperature: f32,
    cache: &'a canvas::Cache,
    surface_cache: &'a canvas::Cache,
}

impl<'a> canvas::Program<Message> for LandscapeCanvas<'a> {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let size = bounds.size();

        // Draw surface (cached)
        let surface = self.surface_cache.draw(renderer, size, |frame| {
            draw_energy_surface(frame, size, self.landscape, 100);
            draw_axes(frame, size);
        });

        // Draw particles (not cached)
        let particles = self.cache.draw(renderer, size, |frame| {
            let color = temperature_to_color(self.temperature, T_END, T_START);
            draw_particles_with_glow(
                frame,
                size,
                self.particles_x,
                self.particles_y,
                color,
                self.temperature,
            );
        });

        vec![surface, particles]
    }
}

/// Draw particles with temperature-based glow
fn draw_particles_with_glow(
    frame: &mut canvas::Frame,
    size: Size,
    xs: &[f32],
    ys: &[f32],
    color: Color,
    temperature: f32,
) {
    let particle_radius = 4.0;
    let t_norm = ((temperature.ln() - T_END.ln()) / (T_START.ln() - T_END.ln())).clamp(0.0, 1.0);

    for (x, y) in xs.iter().zip(ys.iter()) {
        let sx = (*x - X_MIN) / (X_MAX - X_MIN) * size.width;
        let sy = (*y - Y_MIN) / (Y_MAX - Y_MIN) * size.height;

        if sx >= 0.0 && sx <= size.width && sy >= 0.0 && sy <= size.height {
            // Glow radius increases with temperature
            let glow_radius = particle_radius * (1.5 + t_norm);
            let glow = canvas::Path::circle(Point::new(sx, sy), glow_radius);
            frame.fill(&glow, Color::from_rgba(1.0, 1.0, 1.0, 0.2));

            let circle = canvas::Path::circle(Point::new(sx, sy), particle_radius);
            frame.fill(&circle, color);
        }
    }
}

fn main() -> iced::Result {
    iced::application(App::new, update, view)
        .subscription(subscription)
        .title("Loss Landscape Visualizer")
        .antialiasing(true)
        .run()
}
