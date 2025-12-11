//! Annealing Demonstration
//!
//! Shows a single particle system transitioning through all three computational modes
//! by gradually lowering temperature:
//!
//! Phase 1 (ENTROPY):   T = 10.0 → 1.0    Chaotic exploration, extract randomness
//! Phase 2 (SAMPLE):    T = 1.0 → 0.01    Sample posterior, find both optima
//! Phase 3 (OPTIMIZE):  T = 0.01 → 0.0001 Converge to minimum
//!
//! This demonstrates that optimization, sampling, and entropy generation
//! are ONE algorithm at different temperatures.
//!
//! Run with: cargo run --release --features "viz gpu" --bin annealing-demo

use iced::keyboard;
use iced::mouse;
use iced::widget::{canvas, container, Column, Text};
use iced::{event, Color, Element, Event, Length, Point, Rectangle, Renderer, Subscription, Theme};
use nbody_entropy::thermodynamic::{ThermodynamicMode, ThermodynamicParticle, ThermodynamicSystem};

const PARTICLE_COUNT: usize = 500;
const DIM: usize = 2;

// Annealing schedule
const PHASE1_STEPS: u32 = 500;   // Entropy phase
const PHASE2_STEPS: u32 = 1000;  // Sampling phase
const PHASE3_STEPS: u32 = 500;   // Optimization phase
const TOTAL_STEPS: u32 = PHASE1_STEPS + PHASE2_STEPS + PHASE3_STEPS;

// Temperature bounds
const T_START: f32 = 10.0;
const T_SAMPLE: f32 = 0.1;
const T_END: f32 = 0.0001;

// Domain
const DOMAIN_MIN: f32 = -4.0;
const DOMAIN_MAX: f32 = 4.0;
const DOMAIN_SIZE: f32 = DOMAIN_MAX - DOMAIN_MIN;

fn main() -> iced::Result {
    iced::application(App::new, update, view)
        .subscription(subscription)
        .title("Annealing Demo - Unified Thermodynamic Computation")
        .run()
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    Entropy,
    Sample,
    Optimize,
    Complete,
}

impl Phase {
    fn name(&self) -> &'static str {
        match self {
            Phase::Entropy => "ENTROPY",
            Phase::Sample => "SAMPLE",
            Phase::Optimize => "OPTIMIZE",
            Phase::Complete => "COMPLETE",
        }
    }

    fn color(&self) -> Color {
        match self {
            Phase::Entropy => Color::from_rgb(1.0, 0.4, 0.2),
            Phase::Sample => Color::from_rgb(0.2, 0.8, 0.4),
            Phase::Optimize => Color::from_rgb(0.2, 0.6, 1.0),
            Phase::Complete => Color::from_rgb(0.8, 0.8, 0.8),
        }
    }
}

struct App {
    system: ThermodynamicSystem,
    particles: Vec<ThermodynamicParticle>,
    cache: canvas::Cache,
    step_count: u32,
    temperature: f32,
    phase: Phase,
    paused: bool,
    // Statistics history
    mean_energy_history: Vec<f32>,
    min_energy_history: Vec<f32>,
    temperature_history: Vec<f32>,
    // Entropy collected during phase 1
    entropy_bits_collected: usize,
    // Final results
    particles_at_opt1: usize,
    particles_at_opt2: usize,
}

impl App {
    fn new() -> Self {
        let temperature = T_START;
        let system = ThermodynamicSystem::new(PARTICLE_COUNT, DIM, temperature);
        let particles = system.read_particles();

        Self {
            system,
            particles,
            cache: canvas::Cache::new(),
            step_count: 0,
            temperature,
            phase: Phase::Entropy,
            paused: false,
            mean_energy_history: Vec::new(),
            min_energy_history: Vec::new(),
            temperature_history: Vec::new(),
            entropy_bits_collected: 0,
            particles_at_opt1: 0,
            particles_at_opt2: 0,
        }
    }

    fn reset(&mut self) {
        self.system = ThermodynamicSystem::new(PARTICLE_COUNT, DIM, T_START);
        self.particles = self.system.read_particles();
        self.step_count = 0;
        self.temperature = T_START;
        self.phase = Phase::Entropy;
        self.mean_energy_history.clear();
        self.min_energy_history.clear();
        self.temperature_history.clear();
        self.entropy_bits_collected = 0;
        self.particles_at_opt1 = 0;
        self.particles_at_opt2 = 0;
    }

    fn compute_temperature(&self) -> f32 {
        if self.step_count < PHASE1_STEPS {
            // Phase 1: T_START → 1.0 (log scale)
            let t = self.step_count as f32 / PHASE1_STEPS as f32;
            let log_start = T_START.ln();
            let log_end = 1.0_f32.ln();
            (log_start + t * (log_end - log_start)).exp()
        } else if self.step_count < PHASE1_STEPS + PHASE2_STEPS {
            // Phase 2: 1.0 → T_SAMPLE (log scale)
            let t = (self.step_count - PHASE1_STEPS) as f32 / PHASE2_STEPS as f32;
            let log_start = 1.0_f32.ln();
            let log_end = T_SAMPLE.ln();
            (log_start + t * (log_end - log_start)).exp()
        } else if self.step_count < TOTAL_STEPS {
            // Phase 3: T_SAMPLE → T_END (log scale)
            let t = (self.step_count - PHASE1_STEPS - PHASE2_STEPS) as f32 / PHASE3_STEPS as f32;
            let log_start = T_SAMPLE.ln();
            let log_end = T_END.ln();
            (log_start + t * (log_end - log_start)).exp()
        } else {
            T_END
        }
    }

    fn compute_phase(&self) -> Phase {
        if self.step_count < PHASE1_STEPS {
            Phase::Entropy
        } else if self.step_count < PHASE1_STEPS + PHASE2_STEPS {
            Phase::Sample
        } else if self.step_count < TOTAL_STEPS {
            Phase::Optimize
        } else {
            Phase::Complete
        }
    }

    fn count_particles_at_optima(&self) -> (usize, usize) {
        let opt1 = (1.5_f32, 2.0_f32);
        let opt2 = (-1.5_f32, -2.0_f32);
        let tolerance = 0.5;

        let mut count1 = 0;
        let mut count2 = 0;

        for p in &self.particles {
            if p.pos[0].is_nan() { continue; }
            let d1 = ((p.pos[0] - opt1.0).powi(2) + (p.pos[1] - opt1.1).powi(2)).sqrt();
            let d2 = ((p.pos[0] - opt2.0).powi(2) + (p.pos[1] - opt2.1).powi(2)).sqrt();
            if d1 < tolerance { count1 += 1; }
            if d2 < tolerance { count2 += 1; }
        }

        (count1, count2)
    }
}

#[derive(Debug, Clone)]
enum Message {
    Tick,
    TogglePause,
    Reset,
    Event(Event),
}

fn update(app: &mut App, message: Message) {
    match message {
        Message::Tick => {
            if app.paused || app.phase == Phase::Complete {
                return;
            }

            // Update temperature
            app.temperature = app.compute_temperature();
            app.system.set_temperature(app.temperature);

            // Run simulation step
            app.system.step();
            app.step_count += 1;

            // Update phase
            app.phase = app.compute_phase();

            // Read particles
            app.particles = app.system.read_particles();

            // Collect entropy in phase 1
            if app.phase == Phase::Entropy {
                let entropy = app.system.extract_entropy();
                app.entropy_bits_collected += entropy.len() * 32;
            }

            // Record statistics (every 10 steps to avoid too much data)
            if app.step_count % 10 == 0 {
                let energies: Vec<f32> = app.particles.iter()
                    .filter(|p| !p.energy.is_nan())
                    .map(|p| p.energy)
                    .collect();

                if !energies.is_empty() {
                    let mean = energies.iter().sum::<f32>() / energies.len() as f32;
                    let min = energies.iter().cloned().fold(f32::MAX, f32::min);
                    app.mean_energy_history.push(mean);
                    app.min_energy_history.push(min);
                    app.temperature_history.push(app.temperature);
                }
            }

            // Update optima counts
            let (c1, c2) = app.count_particles_at_optima();
            app.particles_at_opt1 = c1;
            app.particles_at_opt2 = c2;

            app.cache.clear();
        }
        Message::TogglePause => {
            app.paused = !app.paused;
        }
        Message::Reset => {
            app.reset();
        }
        Message::Event(Event::Keyboard(keyboard::Event::KeyPressed { key, .. })) => {
            match key {
                keyboard::Key::Named(keyboard::key::Named::Space) => {
                    app.paused = !app.paused;
                }
                keyboard::Key::Character(c) if c.as_str() == "r" => {
                    app.reset();
                }
                _ => {}
            }
        }
        Message::Event(_) => {}
    }
}

fn view(app: &App) -> Element<'_, Message> {
    let progress = (app.step_count as f32 / TOTAL_STEPS as f32 * 100.0).min(100.0);

    let status = if app.paused {
        "PAUSED"
    } else if app.phase == Phase::Complete {
        "COMPLETE"
    } else {
        "RUNNING"
    };

    let info_text = format!(
        "Phase: {} | T: {:.6} | Step: {}/{} ({:.0}%) | Status: {}",
        app.phase.name(),
        app.temperature,
        app.step_count,
        TOTAL_STEPS,
        progress,
        status
    );

    let stats_text = format!(
        "Near Opt1: {} | Near Opt2: {} | Entropy bits: {}",
        app.particles_at_opt1,
        app.particles_at_opt2,
        app.entropy_bits_collected
    );

    let controls_text = "Space=Pause  R=Reset";

    Column::new()
        .push(Text::new(info_text).size(16))
        .push(Text::new(stats_text).size(14))
        .push(Text::new(controls_text).size(12))
        .push(
            container(
                canvas(app)
                    .width(Length::Fill)
                    .height(Length::Fill),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_| container::Style {
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
            let phase_color = self.phase.color();

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

            // Draw optima markers
            for (w1, w2) in [(1.5, 2.0), (-1.5, -2.0)] {
                let opt_x = ((w1 - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
                let opt_y = ((w2 - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;
                let opt = canvas::Path::circle(Point::new(opt_x, opt_y), 15.0);
                frame.stroke(
                    &opt,
                    canvas::Stroke::default()
                        .with_color(Color::from_rgba(0.0, 1.0, 0.5, 0.8))
                        .with_width(3.0),
                );
            }

            // Draw particles
            let max_energy = 5.0;
            for p in &self.particles {
                if p.pos[0].is_nan() || p.pos[1].is_nan() || p.energy.is_nan() {
                    continue;
                }
                let x = ((p.pos[0] - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
                let y = ((p.pos[1] - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;
                if x.is_nan() || y.is_nan() {
                    continue;
                }

                let t = (p.energy / max_energy).clamp(0.0, 1.0);

                // Blend particle color with phase color
                let base_brightness = 1.0 - t * 0.5;
                let color = Color::from_rgb(
                    phase_color.r * base_brightness,
                    phase_color.g * base_brightness,
                    phase_color.b * base_brightness,
                );

                let circle = canvas::Path::circle(Point::new(x, y), 3.0);
                frame.fill(&circle, color);
            }

            // Draw phase progress bar at bottom
            let bar_height = 30.0;
            let bar_y = size.height - bar_height - 10.0;

            // Background
            let bg = canvas::Path::rectangle(
                Point::new(10.0, bar_y),
                iced::Size::new(size.width - 20.0, bar_height),
            );
            frame.fill(&bg, Color::from_rgba(0.1, 0.1, 0.1, 0.8));

            // Phase sections
            let phase_width = (size.width - 20.0) / 3.0;

            // Entropy section
            let entropy_fill = if self.step_count < PHASE1_STEPS {
                self.step_count as f32 / PHASE1_STEPS as f32
            } else { 1.0 };
            let entropy_bar = canvas::Path::rectangle(
                Point::new(10.0, bar_y),
                iced::Size::new(phase_width * entropy_fill, bar_height),
            );
            frame.fill(&entropy_bar, Phase::Entropy.color());

            // Sample section
            let sample_fill = if self.step_count < PHASE1_STEPS {
                0.0
            } else if self.step_count < PHASE1_STEPS + PHASE2_STEPS {
                (self.step_count - PHASE1_STEPS) as f32 / PHASE2_STEPS as f32
            } else { 1.0 };
            let sample_bar = canvas::Path::rectangle(
                Point::new(10.0 + phase_width, bar_y),
                iced::Size::new(phase_width * sample_fill, bar_height),
            );
            frame.fill(&sample_bar, Phase::Sample.color());

            // Optimize section
            let opt_fill = if self.step_count < PHASE1_STEPS + PHASE2_STEPS {
                0.0
            } else if self.step_count < TOTAL_STEPS {
                (self.step_count - PHASE1_STEPS - PHASE2_STEPS) as f32 / PHASE3_STEPS as f32
            } else { 1.0 };
            let opt_bar = canvas::Path::rectangle(
                Point::new(10.0 + phase_width * 2.0, bar_y),
                iced::Size::new(phase_width * opt_fill, bar_height),
            );
            frame.fill(&opt_bar, Phase::Optimize.color());

            // Phase dividers
            for i in 1..3 {
                let x = 10.0 + phase_width * i as f32;
                let divider = canvas::Path::line(
                    Point::new(x, bar_y),
                    Point::new(x, bar_y + bar_height),
                );
                frame.stroke(&divider, canvas::Stroke::default().with_color(Color::WHITE).with_width(2.0));
            }

            // Draw energy history graph (top-right)
            if !self.mean_energy_history.is_empty() {
                let graph_width = 200.0;
                let graph_height = 100.0;
                let graph_x = size.width - graph_width - 20.0;
                let graph_y = 20.0;

                // Background
                let graph_bg = canvas::Path::rectangle(
                    Point::new(graph_x - 5.0, graph_y - 5.0),
                    iced::Size::new(graph_width + 10.0, graph_height + 10.0),
                );
                frame.fill(&graph_bg, Color::from_rgba(0.0, 0.0, 0.0, 0.7));

                // Plot mean energy
                let max_e = self.mean_energy_history.iter().cloned().fold(0.1_f32, f32::max);
                let len = self.mean_energy_history.len();

                for i in 1..len {
                    let x1 = graph_x + ((i - 1) as f32 / len as f32) * graph_width;
                    let x2 = graph_x + (i as f32 / len as f32) * graph_width;
                    let y1 = graph_y + graph_height - (self.mean_energy_history[i-1] / max_e).min(1.0) * graph_height;
                    let y2 = graph_y + graph_height - (self.mean_energy_history[i] / max_e).min(1.0) * graph_height;

                    if !y1.is_nan() && !y2.is_nan() {
                        let line = canvas::Path::line(Point::new(x1, y1), Point::new(x2, y2));
                        frame.stroke(&line, canvas::Stroke::default().with_color(Color::from_rgb(1.0, 1.0, 0.0)).with_width(1.5));
                    }
                }

                // Plot min energy
                for i in 1..len {
                    let x1 = graph_x + ((i - 1) as f32 / len as f32) * graph_width;
                    let x2 = graph_x + (i as f32 / len as f32) * graph_width;
                    let y1 = graph_y + graph_height - (self.min_energy_history[i-1] / max_e).min(1.0) * graph_height;
                    let y2 = graph_y + graph_height - (self.min_energy_history[i] / max_e).min(1.0) * graph_height;

                    if !y1.is_nan() && !y2.is_nan() {
                        let line = canvas::Path::line(Point::new(x1, y1), Point::new(x2, y2));
                        frame.stroke(&line, canvas::Stroke::default().with_color(Color::from_rgb(0.0, 1.0, 1.0)).with_width(1.5));
                    }
                }
            }

            // Draw temperature history (below energy graph)
            if !self.temperature_history.is_empty() {
                let graph_width = 200.0;
                let graph_height = 60.0;
                let graph_x = size.width - graph_width - 20.0;
                let graph_y = 140.0;

                // Background
                let graph_bg = canvas::Path::rectangle(
                    Point::new(graph_x - 5.0, graph_y - 5.0),
                    iced::Size::new(graph_width + 10.0, graph_height + 10.0),
                );
                frame.fill(&graph_bg, Color::from_rgba(0.0, 0.0, 0.0, 0.7));

                // Plot temperature (log scale)
                let len = self.temperature_history.len();
                let log_max = T_START.ln();
                let log_min = T_END.ln();

                for i in 1..len {
                    let x1 = graph_x + ((i - 1) as f32 / len as f32) * graph_width;
                    let x2 = graph_x + (i as f32 / len as f32) * graph_width;

                    let t1 = self.temperature_history[i-1].max(T_END);
                    let t2 = self.temperature_history[i].max(T_END);

                    let y1 = graph_y + graph_height - ((t1.ln() - log_min) / (log_max - log_min)).clamp(0.0, 1.0) * graph_height;
                    let y2 = graph_y + graph_height - ((t2.ln() - log_min) / (log_max - log_min)).clamp(0.0, 1.0) * graph_height;

                    if !y1.is_nan() && !y2.is_nan() {
                        let line = canvas::Path::line(Point::new(x1, y1), Point::new(x2, y2));
                        frame.stroke(&line, canvas::Stroke::default().with_color(Color::from_rgb(1.0, 0.5, 0.0)).with_width(2.0));
                    }
                }
            }
        });

        vec![geometry]
    }
}
