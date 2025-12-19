//! Parallel Tempering Visualizer
//!
//! Real-time visualization of replica exchange Monte Carlo on 2D energy surfaces.
//! Multiple replicas run at different temperatures, periodically exchanging states.
//!
//! Hot replicas explore globally, cold replicas exploit locally.
//! Watch how information flows from hot to cold through exchanges!
//!
//! Controls:
//!   Space = Pause/Resume
//!   R = Reset
//!   L = Switch landscape
//!   S = Force swap attempt
//!   +/- = Speed up/down
//!
//! Run with: cargo run --release --features "viz gpu" --bin parallel-tempering-viz

use iced::keyboard;
use iced::mouse;
use iced::widget::{Column, Text, canvas, container};
use iced::{
    Color, Element, Event, Length, Point, Rectangle, Renderer, Size, Subscription, Theme, event,
};
use temper::ThermodynamicSystem;
use temper::viz::{Landscape, X_MAX, X_MIN, Y_MAX, Y_MIN, energy_to_color};

const PARTICLES_PER_REPLICA: usize = 25;
const NUM_REPLICAS: usize = 4;
const SWAP_INTERVAL: u32 = 50; // Attempt swap every N steps

// Temperature ladder (geometric spacing)
const TEMPERATURES: [f32; NUM_REPLICAS] = [0.01, 0.1, 0.5, 2.0];

struct Replica {
    system: ThermodynamicSystem,
    temperature: f32,
    particles_x: Vec<f32>,
    particles_y: Vec<f32>,
    min_energy: f32,
    mean_energy: f32,
}

impl Replica {
    fn new(landscape: Landscape, temperature: f32) -> Self {
        let system =
            ThermodynamicSystem::with_expr(PARTICLES_PER_REPLICA, 2, temperature, landscape.expr());
        Self {
            system,
            temperature,
            particles_x: vec![0.0; PARTICLES_PER_REPLICA],
            particles_y: vec![0.0; PARTICLES_PER_REPLICA],
            min_energy: 0.0,
            mean_energy: 0.0,
        }
    }

    fn step(&mut self) {
        self.system.set_temperature(self.temperature);
        self.system.step();
    }

    fn read_particles(&mut self) {
        let particles = self.system.read_particles();
        let mut sum_energy = 0.0;
        let mut min_energy = f32::MAX;
        let mut count = 0;

        for (i, p) in particles.iter().enumerate() {
            if p.energy.is_finite() {
                self.particles_x[i] = p.pos[0].to_f32();
                self.particles_y[i] = p.pos[1].to_f32();
                sum_energy += p.energy;
                if p.energy < min_energy {
                    min_energy = p.energy;
                }
                count += 1;
            }
        }

        self.min_energy = min_energy;
        self.mean_energy = if count > 0 {
            sum_energy / count as f32
        } else {
            0.0
        };
    }
}

struct App {
    replicas: Vec<Replica>,
    landscape: Landscape,
    step_count: u32,
    paused: bool,
    speed: u32,
    swap_count: u32,
    last_swap: Option<(usize, usize)>, // Which replicas swapped
    swap_flash: u32,                   // Counter for visual flash
    cache: canvas::Cache,
    surface_cache: canvas::Cache,
    rng_seed: u64,
}

impl App {
    fn new() -> (Self, iced::Task<Message>) {
        let landscape = Landscape::Rastrigin;
        let replicas = TEMPERATURES
            .iter()
            .map(|&t| Replica::new(landscape, t))
            .collect();

        let mut buf = [0u8; 8];
        getrandom::fill(&mut buf).unwrap();
        let rng_seed = u64::from_le_bytes(buf);

        (
            Self {
                replicas,
                landscape,
                step_count: 0,
                paused: false,
                speed: 1,
                swap_count: 0,
                last_swap: None,
                swap_flash: 0,
                cache: canvas::Cache::new(),
                surface_cache: canvas::Cache::new(),
                rng_seed,
            },
            iced::Task::none(),
        )
    }

    fn reset(&mut self) {
        self.replicas = TEMPERATURES
            .iter()
            .map(|&t| Replica::new(self.landscape, t))
            .collect();
        self.step_count = 0;
        self.swap_count = 0;
        self.last_swap = None;
        self.swap_flash = 0;
        self.cache.clear();
    }

    fn switch_landscape(&mut self) {
        self.landscape = self.landscape.next();
        self.surface_cache.clear();
        self.reset();
    }

    /// Attempt replica exchange between adjacent temperature levels
    fn attempt_swaps(&mut self) {
        // Try swapping adjacent pairs
        for i in 0..(NUM_REPLICAS - 1) {
            let t_i = self.replicas[i].temperature;
            let t_j = self.replicas[i + 1].temperature;
            let e_i = self.replicas[i].mean_energy;
            let e_j = self.replicas[i + 1].mean_energy;

            // Metropolis criterion for replica exchange
            // Accept if delta < 0, else accept with probability exp(-delta)
            let delta = (1.0 / t_i - 1.0 / t_j) * (e_j - e_i);

            let accept = if delta <= 0.0 {
                true
            } else {
                // Generate random number for acceptance
                self.rng_seed = self
                    .rng_seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1);
                let rand = (self.rng_seed >> 33) as f32 / (u32::MAX >> 1) as f32;
                rand < (-delta).exp()
            };

            if accept {
                // Swap the particle positions between replicas
                self.swap_replicas(i, i + 1);
                self.swap_count += 1;
                self.last_swap = Some((i, i + 1));
                self.swap_flash = 30; // Flash for 30 frames
            }
        }
    }

    fn swap_replicas(&mut self, i: usize, j: usize) {
        // Read particles from both replicas
        let particles_i = self.replicas[i].system.read_particles();
        let particles_j = self.replicas[j].system.read_particles();

        let expr = self.landscape.expr();

        // Create new systems with swapped positions
        let new_system_i = ThermodynamicSystem::with_expr(
            PARTICLES_PER_REPLICA,
            2,
            self.replicas[i].temperature,
            expr.clone(),
        );
        let new_system_j = ThermodynamicSystem::with_expr(
            PARTICLES_PER_REPLICA,
            2,
            self.replicas[j].temperature,
            expr,
        );

        // Copy positions from j to i and vice versa
        for k in 0..PARTICLES_PER_REPLICA {
            self.replicas[i].particles_x[k] = particles_j[k].pos[0].to_f32();
            self.replicas[i].particles_y[k] = particles_j[k].pos[1].to_f32();
            self.replicas[j].particles_x[k] = particles_i[k].pos[0].to_f32();
            self.replicas[j].particles_y[k] = particles_i[k].pos[1].to_f32();
        }

        self.replicas[i].system = new_system_i;
        self.replicas[j].system = new_system_j;
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
                    // Step all replicas
                    for replica in &mut app.replicas {
                        replica.step();
                    }

                    app.step_count += 1;

                    // Attempt swaps periodically
                    if app.step_count % SWAP_INTERVAL == 0 {
                        // Read particles first to get current energies
                        for replica in &mut app.replicas {
                            replica.read_particles();
                        }
                        app.attempt_swaps();
                    }
                }

                // Read particles for display
                for replica in &mut app.replicas {
                    replica.read_particles();
                }

                // Decay swap flash
                if app.swap_flash > 0 {
                    app.swap_flash -= 1;
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
                    keyboard::Key::Character("l") => {
                        app.switch_landscape();
                    }
                    keyboard::Key::Character("s") => {
                        // Force swap attempt
                        for replica in &mut app.replicas {
                            replica.read_particles();
                        }
                        app.attempt_swaps();
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
    let status = if app.paused { "PAUSED" } else { "RUNNING" };

    let title = format!(
        "Parallel Tempering: {} | {} | Swaps: {}",
        app.landscape.name(),
        status,
        app.swap_count
    );

    let swap_info = if let Some((i, j)) = app.last_swap {
        if app.swap_flash > 0 {
            format!(
                "SWAP! T={:.2} <-> T={:.2}",
                TEMPERATURES[i], TEMPERATURES[j]
            )
        } else {
            format!(
                "Last: T={:.2} <-> T={:.2}",
                TEMPERATURES[i], TEMPERATURES[j]
            )
        }
    } else {
        "No swaps yet".to_string()
    };

    let controls = "Space=Pause  R=Reset  L=Landscape  S=Force Swap  +/-=Speed";

    // Build temperature labels
    let temp_labels: Vec<String> = app
        .replicas
        .iter()
        .map(|r| format!("T={:.2} E={:.2}", r.temperature, r.min_energy))
        .collect();

    Column::new()
        .push(Text::new(title).size(18))
        .push(Text::new(app.landscape.description()).size(12))
        .push(Text::new(swap_info).size(14).color(if app.swap_flash > 0 {
            Color::from_rgb(1.0, 0.5, 0.0)
        } else {
            Color::WHITE
        }))
        .push(Text::new(controls).size(12))
        .push(Text::new(temp_labels.join("  |  ")).size(11))
        .push(
            container(
                canvas(ParallelTemperingCanvas {
                    landscape: &app.landscape,
                    replicas: &app.replicas,
                    last_swap: app.last_swap,
                    swap_flash: app.swap_flash,
                    cache: &app.cache,
                    surface_cache: &app.surface_cache,
                })
                .width(Length::Fixed(800.0))
                .height(Length::Fixed(220.0)),
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

struct ParallelTemperingCanvas<'a> {
    landscape: &'a Landscape,
    replicas: &'a [Replica],
    last_swap: Option<(usize, usize)>,
    swap_flash: u32,
    cache: &'a canvas::Cache,
    surface_cache: &'a canvas::Cache,
}

impl<'a> canvas::Program<Message> for ParallelTemperingCanvas<'a> {
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
        let panel_width = size.width / NUM_REPLICAS as f32;
        let panel_size = Size::new(panel_width - 10.0, size.height - 10.0);

        // Draw surface once (cached)
        let surface = self.surface_cache.draw(renderer, size, |frame| {
            for i in 0..NUM_REPLICAS {
                let offset_x = i as f32 * panel_width + 5.0;
                draw_energy_surface_panel(frame, panel_size, offset_x, 5.0, self.landscape);

                // Draw temperature label background
                let label_rect =
                    canvas::Path::rectangle(Point::new(offset_x, 5.0), Size::new(60.0, 16.0));
                frame.fill(&label_rect, Color::from_rgba(0.0, 0.0, 0.0, 0.7));
            }
        });

        // Draw particles (not cached)
        let particles = self.cache.draw(renderer, size, |frame| {
            for (i, replica) in self.replicas.iter().enumerate() {
                let offset_x = i as f32 * panel_width + 5.0;

                // Highlight if this replica was involved in a swap
                let is_swapping = if let Some((a, b)) = self.last_swap {
                    self.swap_flash > 0 && (i == a || i == b)
                } else {
                    false
                };

                if is_swapping {
                    let highlight = canvas::Path::rectangle(
                        Point::new(offset_x - 2.0, 3.0),
                        Size::new(panel_size.width + 4.0, panel_size.height + 4.0),
                    );
                    let alpha = self.swap_flash as f32 / 30.0;
                    frame.stroke(
                        &highlight,
                        canvas::Stroke::default()
                            .with_color(Color::from_rgba(1.0, 0.5, 0.0, alpha))
                            .with_width(3.0),
                    );
                }

                draw_particles_panel(
                    frame,
                    panel_size,
                    offset_x,
                    5.0,
                    &replica.particles_x,
                    &replica.particles_y,
                    replica.temperature,
                );
            }
        });

        vec![surface, particles]
    }
}

fn draw_energy_surface_panel(
    frame: &mut canvas::Frame,
    size: Size,
    offset_x: f32,
    offset_y: f32,
    landscape: &Landscape,
) {
    let resolution = 40; // Lower res for multiple panels
    let cell_w = size.width / resolution as f32;
    let cell_h = size.height / resolution as f32;

    // Find energy range
    let mut min_e = f32::MAX;
    let mut max_e = f32::MIN;

    for iy in 0..resolution {
        for ix in 0..resolution {
            let x = X_MIN + (ix as f32 + 0.5) / resolution as f32 * (X_MAX - X_MIN);
            let y = Y_MIN + (iy as f32 + 0.5) / resolution as f32 * (Y_MAX - Y_MIN);
            let e = landscape.energy(x, y);
            if e.is_finite() {
                min_e = min_e.min(e);
                max_e = max_e.max(e);
            }
        }
    }

    let log_min = min_e.max(0.001).ln();
    let log_max = max_e.max(0.01).ln();

    for iy in 0..resolution {
        for ix in 0..resolution {
            let x = X_MIN + (ix as f32 + 0.5) / resolution as f32 * (X_MAX - X_MIN);
            let y = Y_MIN + (iy as f32 + 0.5) / resolution as f32 * (Y_MAX - Y_MIN);
            let e = landscape.energy(x, y);

            let norm = if e.is_finite() && log_max > log_min {
                ((e.max(0.001).ln() - log_min) / (log_max - log_min)).clamp(0.0, 1.0)
            } else {
                1.0
            };

            let color = energy_to_color(norm);

            let rect = canvas::Path::rectangle(
                Point::new(offset_x + ix as f32 * cell_w, offset_y + iy as f32 * cell_h),
                Size::new(cell_w + 0.5, cell_h + 0.5),
            );
            frame.fill(&rect, color);
        }
    }
}

fn draw_particles_panel(
    frame: &mut canvas::Frame,
    size: Size,
    offset_x: f32,
    offset_y: f32,
    xs: &[f32],
    ys: &[f32],
    temperature: f32,
) {
    let particle_radius = 3.0;

    // Color based on temperature
    let t_norm =
        ((temperature.ln() - 0.01_f32.ln()) / (2.0_f32.ln() - 0.01_f32.ln())).clamp(0.0, 1.0);

    for (x, y) in xs.iter().zip(ys.iter()) {
        let sx = offset_x + (*x - X_MIN) / (X_MAX - X_MIN) * size.width;
        let sy = offset_y + (*y - Y_MIN) / (Y_MAX - Y_MIN) * size.height;

        if sx >= offset_x
            && sx <= offset_x + size.width
            && sy >= offset_y
            && sy <= offset_y + size.height
        {
            // Glow
            let glow = canvas::Path::circle(Point::new(sx, sy), particle_radius * 1.5);
            frame.fill(&glow, Color::from_rgba(1.0, 1.0, 1.0, 0.3));

            // Particle
            let circle = canvas::Path::circle(Point::new(sx, sy), particle_radius);
            let color = Color::from_rgb(0.3 + 0.7 * t_norm, 0.9 - 0.6 * t_norm, 1.0 - 0.8 * t_norm);
            frame.fill(&circle, color);
        }
    }
}

fn main() -> iced::Result {
    iced::application(App::new, update, view)
        .subscription(subscription)
        .title("Parallel Tempering Visualizer")
        .antialiasing(true)
        .run()
}
