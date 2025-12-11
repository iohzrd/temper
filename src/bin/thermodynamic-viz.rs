//! Unified Thermodynamic Particle System Visualization
//!
//! Interactive demo showing how temperature controls the mode:
//! - T >> 1.0 : ENTROPY mode (chaotic, extract randomness)
//! - T ~ 0.1  : SAMPLE mode (Bayesian inference, SVGD)
//! - T → 0   : OPTIMIZE mode (gradient descent)
//!
//! Controls:
//! - Up/Down arrows: Adjust temperature
//! - Space: Reset particles
//!
//! Run with: cargo run --release --features "viz gpu" --bin thermodynamic-viz

use iced::keyboard;
use iced::mouse;
use iced::widget::{canvas, container, Column, Row, Text};
use iced::{event, Color, Element, Event, Length, Point, Rectangle, Renderer, Subscription, Theme};
use nbody_entropy::thermodynamic::{ThermodynamicMode, ThermodynamicParticle, ThermodynamicSystem};

const PARTICLE_COUNT: usize = 500;
const DIM: usize = 2; // 2D for visualization

// Temperature presets
const T_OPTIMIZE: f32 = 0.001;
const T_SAMPLE: f32 = 0.1;
const T_ENTROPY: f32 = 5.0;

// Domain
const DOMAIN_MIN: f32 = -4.0;
const DOMAIN_MAX: f32 = 4.0;
const DOMAIN_SIZE: f32 = DOMAIN_MAX - DOMAIN_MIN;

fn main() -> iced::Result {
    iced::application(App::new, update, view)
        .subscription(subscription)
        .title("Thermodynamic Computing - Temperature Continuum")
        .run()
}

struct App {
    system: ThermodynamicSystem,
    particles: Vec<ThermodynamicParticle>,
    cache: canvas::Cache,
    step_count: u32,
    temperature: f32,
    entropy_bits: Vec<u32>,
    entropy_history: Vec<f32>, // For showing entropy quality over time
}

impl App {
    fn new() -> Self {
        let temperature = T_SAMPLE; // Start in sampling mode
        let system = ThermodynamicSystem::new(PARTICLE_COUNT, DIM, temperature);
        let particles = system.read_particles();

        Self {
            system,
            particles,
            cache: canvas::Cache::new(),
            step_count: 0,
            temperature,
            entropy_bits: Vec::new(),
            entropy_history: Vec::new(),
        }
    }

    fn reset_particles(&mut self) {
        self.system = ThermodynamicSystem::new(PARTICLE_COUNT, DIM, self.temperature);
        self.particles = self.system.read_particles();
        self.step_count = 0;
        self.entropy_bits.clear();
        self.entropy_history.clear();
    }
}

#[derive(Debug, Clone)]
enum Message {
    Tick,
    IncreaseTemp,
    DecreaseTemp,
    SetMode(ThermodynamicMode),
    Reset,
    Event(Event),
}

fn update(app: &mut App, message: Message) {
    match message {
        Message::Tick => {
            // Run simulation steps
            for _ in 0..2 {
                app.system.step();
            }
            app.step_count += 2;

            // Read particles
            app.particles = app.system.read_particles();

            // In entropy mode, extract entropy
            if app.system.mode() == ThermodynamicMode::Entropy {
                let entropy = app.system.extract_entropy();
                app.entropy_bits.extend(entropy);

                // Compute bit balance as quality metric
                if app.entropy_bits.len() >= 1000 {
                    let ones: u32 = app.entropy_bits.iter()
                        .take(1000)
                        .map(|x| x.count_ones())
                        .sum();
                    let total = 1000 * 32;
                    let balance = (ones as f32 / total as f32 - 0.5).abs();
                    app.entropy_history.push(balance);
                    if app.entropy_history.len() > 100 {
                        app.entropy_history.remove(0);
                    }
                    // Keep only recent entropy
                    if app.entropy_bits.len() > 10000 {
                        app.entropy_bits.drain(0..5000);
                    }
                }
            }

            app.cache.clear();
        }
        Message::IncreaseTemp => {
            app.temperature = (app.temperature * 1.5).min(100.0);
            app.system.set_temperature(app.temperature);
        }
        Message::DecreaseTemp => {
            app.temperature = (app.temperature / 1.5).max(0.0001);
            app.system.set_temperature(app.temperature);
        }
        Message::SetMode(mode) => {
            app.temperature = match mode {
                ThermodynamicMode::Optimize => T_OPTIMIZE,
                ThermodynamicMode::Sample => T_SAMPLE,
                ThermodynamicMode::Entropy => T_ENTROPY,
            };
            app.system.set_temperature(app.temperature);
        }
        Message::Reset => {
            app.reset_particles();
        }
        Message::Event(Event::Keyboard(keyboard::Event::KeyPressed { key, .. })) => {
            match key {
                keyboard::Key::Named(keyboard::key::Named::ArrowUp) => {
                    app.temperature = (app.temperature * 1.5).min(100.0);
                    app.system.set_temperature(app.temperature);
                }
                keyboard::Key::Named(keyboard::key::Named::ArrowDown) => {
                    app.temperature = (app.temperature / 1.5).max(0.0001);
                    app.system.set_temperature(app.temperature);
                }
                keyboard::Key::Named(keyboard::key::Named::Space) => {
                    app.reset_particles();
                }
                keyboard::Key::Character(c) => {
                    match c.as_str() {
                        "1" => {
                            app.temperature = T_OPTIMIZE;
                            app.system.set_temperature(app.temperature);
                        }
                        "2" => {
                            app.temperature = T_SAMPLE;
                            app.system.set_temperature(app.temperature);
                        }
                        "3" => {
                            app.temperature = T_ENTROPY;
                            app.system.set_temperature(app.temperature);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        Message::Event(_) => {}
    }
}

fn view(app: &App) -> Element<'_, Message> {
    let mode = app.system.mode();
    let mode_color = match mode {
        ThermodynamicMode::Optimize => Color::from_rgb(0.2, 0.6, 1.0),
        ThermodynamicMode::Sample => Color::from_rgb(0.2, 0.8, 0.4),
        ThermodynamicMode::Entropy => Color::from_rgb(1.0, 0.4, 0.2),
    };

    let stats = app.system.statistics();

    let info_text = format!(
        "Mode: {} | T: {:.4} | Step: {} | Mean E: {:.4} | Min E: {:.4} | Low E: {:.1}%",
        mode.name(),
        app.temperature,
        app.step_count,
        stats.mean_energy,
        stats.min_energy,
        stats.low_energy_fraction * 100.0
    );

    let controls_text = "Keys: 1=Optimize  2=Sample  3=Entropy  ↑↓=Adjust T  Space=Reset";

    Column::new()
        .push(
            Text::new(info_text)
                .size(16)
        )
        .push(
            Text::new(controls_text)
                .size(14)
        )
        .push(
            container(
                canvas(app)
                    .width(Length::Fill)
                    .height(Length::Fill),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .style(move |_| container::Style {
                background: Some(iced::Background::Color(Color::from_rgb(0.05, 0.05, 0.1))),
                ..Default::default()
            })
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
        let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
            let size = bounds.size();
            let mode = self.system.mode();

            // Draw grid
            let grid_color = Color::from_rgba(0.2, 0.3, 0.4, 0.3);
            for i in -4..=4 {
                let x = ((i as f32 - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
                let y = ((i as f32 - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;

                let vline = canvas::Path::line(Point::new(x, 0.0), Point::new(x, size.height));
                frame.stroke(&vline, canvas::Stroke::default().with_color(grid_color).with_width(0.5));

                let hline = canvas::Path::line(Point::new(0.0, y), Point::new(size.width, y));
                frame.stroke(&hline, canvas::Stroke::default().with_color(grid_color).with_width(0.5));
            }

            // Draw the two optima (for 2D neural net)
            for (w1, w2) in [(1.5, 2.0), (-1.5, -2.0)] {
                let opt_x = ((w1 - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
                let opt_y = ((w2 - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;
                let opt = canvas::Path::circle(Point::new(opt_x, opt_y), 12.0);
                frame.stroke(
                    &opt,
                    canvas::Stroke::default()
                        .with_color(Color::from_rgba(0.0, 1.0, 0.5, 0.8))
                        .with_width(3.0),
                );
            }

            // Draw particles with mode-dependent coloring
            let max_energy = 5.0;
            for p in &self.particles {
                // Skip particles with NaN positions
                if p.pos[0].is_nan() || p.pos[1].is_nan() || p.energy.is_nan() {
                    continue;
                }
                let x = ((p.pos[0] - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
                let y = ((p.pos[1] - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;
                if x.is_nan() || y.is_nan() {
                    continue;
                }
                let pos = Point::new(x, y);

                let t = (p.energy / max_energy).clamp(0.0, 1.0);

                // Color scheme based on mode
                let color = match mode {
                    ThermodynamicMode::Optimize => {
                        // Blue gradient: bright blue for low energy
                        Color::from_rgb(0.2 + 0.3 * t, 0.4 + 0.2 * (1.0 - t), 0.9 * (1.0 - t) + 0.3)
                    }
                    ThermodynamicMode::Sample => {
                        // Green gradient: green for low energy
                        Color::from_rgb(0.2 + 0.6 * t, 0.8 * (1.0 - t) + 0.2, 0.3)
                    }
                    ThermodynamicMode::Entropy => {
                        // Orange/red gradient: chaotic fire colors
                        Color::from_rgb(0.9 + 0.1 * t, 0.4 * (1.0 - t) + 0.2, 0.1)
                    }
                };

                let radius = 3.0;
                let circle = canvas::Path::circle(pos, radius);
                frame.fill(&circle, color);
            }

            // Draw mode indicator
            let mode_color = match mode {
                ThermodynamicMode::Optimize => Color::from_rgb(0.2, 0.6, 1.0),
                ThermodynamicMode::Sample => Color::from_rgb(0.2, 0.8, 0.4),
                ThermodynamicMode::Entropy => Color::from_rgb(1.0, 0.4, 0.2),
            };

            // Temperature bar on the right side
            let bar_width = 20.0;
            let bar_height = size.height * 0.6;
            let bar_x = size.width - 40.0;
            let bar_y = size.height * 0.2;

            // Background
            let bar_bg = canvas::Path::rectangle(
                Point::new(bar_x, bar_y),
                iced::Size::new(bar_width, bar_height),
            );
            frame.fill(&bar_bg, Color::from_rgba(0.2, 0.2, 0.2, 0.8));

            // Temperature indicator (log scale)
            // Map log(T) from log(0.0001) to log(100) to [0, 1]
            let t_min = 0.0001_f32.ln(); // -9.2
            let t_max = 100.0_f32.ln();  // 4.6
            let t_log = (self.temperature.max(0.0001).ln() - t_min) / (t_max - t_min);
            let t_pos = t_log.clamp(0.0, 1.0);
            let indicator_y = bar_y + bar_height * (1.0 - t_pos);

            let indicator = canvas::Path::rectangle(
                Point::new(bar_x - 5.0, indicator_y - 3.0),
                iced::Size::new(bar_width + 10.0, 6.0),
            );
            frame.fill(&indicator, mode_color);

            // Mode zone indicators (unused for now, commented out to avoid warnings)
            // let _zones = [
            //     ("OPT", t_min, T_OPTIMIZE.ln(), Color::from_rgb(0.2, 0.6, 1.0)),
            //     ("SMP", T_OPTIMIZE.ln(), T_ENTROPY.ln(), Color::from_rgb(0.2, 0.8, 0.4)),
            //     ("ENT", T_ENTROPY.ln(), t_max, Color::from_rgb(1.0, 0.4, 0.2)),
            // ];

            // Draw loss histogram in bottom-left
            let hist_width = 150.0;
            let hist_height = 100.0;
            let hist_x = 20.0;
            let hist_y = size.height - hist_height - 40.0;
            let num_bins = 15;
            let bin_width = hist_width / num_bins as f32;

            let mut bins = vec![0usize; num_bins];
            for p in &self.particles {
                if p.energy.is_nan() {
                    continue;
                }
                let bin_idx = ((p.energy / max_energy) * (num_bins - 1) as f32).clamp(0.0, (num_bins - 1) as f32) as usize;
                bins[bin_idx] += 1;
            }
            let max_count = bins.iter().max().copied().unwrap_or(1).max(1);

            // Histogram background
            let bg = canvas::Path::rectangle(
                Point::new(hist_x - 5.0, hist_y - 5.0),
                iced::Size::new(hist_width + 10.0, hist_height + 10.0),
            );
            frame.fill(&bg, Color::from_rgba(0.0, 0.0, 0.0, 0.7));

            // Histogram bars
            for (i, &count) in bins.iter().enumerate() {
                let bar_height = (count as f32 / max_count as f32) * hist_height;
                let bx = hist_x + i as f32 * bin_width;
                let by = hist_y + hist_height - bar_height;

                let bar = canvas::Path::rectangle(
                    Point::new(bx, by),
                    iced::Size::new(bin_width - 1.0, bar_height),
                );
                frame.fill(&bar, mode_color);
            }

            // In entropy mode, show entropy quality indicator
            if mode == ThermodynamicMode::Entropy && !self.entropy_history.is_empty() {
                let ent_x = hist_x + hist_width + 30.0;
                let ent_width = 100.0;
                let ent_height = 50.0;

                // Background
                let ent_bg = canvas::Path::rectangle(
                    Point::new(ent_x - 5.0, hist_y + hist_height - ent_height - 5.0),
                    iced::Size::new(ent_width + 10.0, ent_height + 10.0),
                );
                frame.fill(&ent_bg, Color::from_rgba(0.0, 0.0, 0.0, 0.7));

                // Draw entropy balance history
                for (i, &balance) in self.entropy_history.iter().enumerate() {
                    let ex = ent_x + (i as f32 / self.entropy_history.len() as f32) * ent_width;
                    let ey = hist_y + hist_height - ent_height / 2.0 - balance * ent_height * 10.0;
                    let dot = canvas::Path::circle(Point::new(ex, ey.clamp(hist_y + hist_height - ent_height, hist_y + hist_height)), 2.0);
                    frame.fill(&dot, Color::from_rgb(0.0, 1.0, 1.0));
                }

                // Reference line at 0 (perfect balance)
                let ref_line = canvas::Path::line(
                    Point::new(ent_x, hist_y + hist_height - ent_height / 2.0),
                    Point::new(ent_x + ent_width, hist_y + hist_height - ent_height / 2.0),
                );
                frame.stroke(&ref_line, canvas::Stroke::default().with_color(Color::from_rgba(0.5, 0.5, 0.5, 0.5)).with_width(1.0));
            }
        });

        vec![geometry]
    }
}
