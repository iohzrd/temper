//! Adaptive Annealing Visualization
//!
//! Compares fixed exponential cooling vs adaptive scheduling side-by-side.
//! Adaptive schedule adjusts based on convergence metrics:
//! - Cools aggressively when already near optimum (convergence detection)
//! - Slows cooling when stuck at high energy
//! - Reheats when trapped in local minima
//!
//! Press 'S' to toggle between Rastrigin and Schwefel functions.
//!
//! Run with: cargo run --release --features "viz gpu" --bin adaptive-annealing

use iced::keyboard;
use iced::mouse;
use iced::widget::{canvas, container, Column, Row, Text};
use iced::{event, Color, Element, Event, Length, Point, Rectangle, Renderer, Subscription, Theme};
use nbody_entropy::thermodynamic::{LossFunction, ThermodynamicParticle, ThermodynamicSystem};

const PARTICLE_COUNT: usize = 500;
const DIM: usize = 2;
const TOTAL_STEPS: u32 = 5000;

// Adaptive parameters
const STALL_WINDOW: usize = 50;
const REHEAT_FACTOR: f32 = 2.0;

#[derive(Clone, Copy, PartialEq)]
enum TestFunction {
    Rastrigin,
    Schwefel,
}

impl TestFunction {
    fn loss_function(&self) -> LossFunction {
        match self {
            Self::Rastrigin => LossFunction::Rastrigin,
            Self::Schwefel => LossFunction::Schwefel,
        }
    }

    fn domain(&self) -> (f32, f32) {
        match self {
            Self::Rastrigin => (-4.0, 4.0),
            Self::Schwefel => (-500.0, 500.0),
        }
    }

    fn t_start(&self) -> f32 {
        match self {
            Self::Rastrigin => 5.0,
            Self::Schwefel => 500.0,
        }
    }

    fn t_end(&self) -> f32 {
        match self {
            Self::Rastrigin => 0.001,
            Self::Schwefel => 0.1,
        }
    }

    fn convergence_threshold(&self) -> f32 {
        match self {
            Self::Rastrigin => 0.1,   // Near zero is converged
            Self::Schwefel => 10.0,   // Near global min (~0) is converged
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Rastrigin => "Rastrigin",
            Self::Schwefel => "Schwefel",
        }
    }

    fn global_opt(&self) -> (f32, f32) {
        match self {
            Self::Rastrigin => (0.0, 0.0),
            Self::Schwefel => (420.97, 420.97),
        }
    }
}

fn main() -> iced::Result {
    iced::application(App::new, update, view)
        .subscription(subscription)
        .title("Adaptive vs Fixed Annealing - Press S to switch function")
        .run()
}

fn rastrigin(x: f32, y: f32) -> f32 {
    let a = 10.0;
    20.0 + (x * x - a * (2.0 * std::f32::consts::PI * x).cos())
        + (y * y - a * (2.0 * std::f32::consts::PI * y).cos())
}

fn schwefel(x: f32, y: f32) -> f32 {
    418.9829 * 2.0 - x * (x.abs().sqrt()).sin() - y * (y.abs().sqrt()).sin()
}

/// Improved adaptive temperature scheduler
struct AdaptiveScheduler {
    temperature: f32,
    t_start: f32,
    t_end: f32,
    base_cooling_rate: f32,
    energy_history: Vec<f32>,
    stall_count: u32,
    pub reheat_count: u32,
    convergence_threshold: f32,
}

impl AdaptiveScheduler {
    fn new(func: TestFunction, total_steps: u32) -> Self {
        let t_start = func.t_start();
        let t_end = func.t_end();
        let base_cooling_rate = (t_end / t_start).powf(1.0 / total_steps as f32);
        Self {
            temperature: t_start,
            t_start,
            t_end,
            base_cooling_rate,
            energy_history: Vec::new(),
            stall_count: 0,
            reheat_count: 0,
            convergence_threshold: func.convergence_threshold(),
        }
    }

    fn update(&mut self, min_energy: f32) -> f32 {
        self.energy_history.push(min_energy);

        // Key improvement #1: Convergence detection
        // If we're already at low energy, cool aggressively to lock in
        if min_energy < self.convergence_threshold {
            self.temperature *= self.base_cooling_rate.powf(2.0);  // 2x faster cooling
            self.stall_count = 0;
            self.temperature = self.temperature.max(self.t_end);
            return self.temperature;
        }

        if self.energy_history.len() < STALL_WINDOW {
            self.temperature *= self.base_cooling_rate;
            return self.temperature;
        }

        let recent_start = self.energy_history.len() - STALL_WINDOW;
        let old_energy = self.energy_history[recent_start];
        let improvement = old_energy - min_energy;

        // Key improvement #2: Relative improvement scaled by energy magnitude
        let improvement_rate = if old_energy > 0.01 {
            improvement / old_energy
        } else {
            improvement
        };

        // Key improvement #3: Different thresholds based on energy level
        let stall_threshold = if min_energy > 100.0 { 0.001 }  // High energy: very sensitive
                             else if min_energy > 10.0 { 0.005 }
                             else if min_energy > 1.0 { 0.01 }
                             else { 0.05 };  // Low energy: less sensitive (allow slow convergence)

        if improvement_rate < stall_threshold && self.temperature > self.t_end * 10.0 {
            self.stall_count += 1;

            // Key improvement #4: Reheat based on energy level, not just stall count
            if self.stall_count > 80 && min_energy > self.convergence_threshold * 10.0 {
                // Very stuck at high energy - reheat!
                self.temperature = (self.temperature * REHEAT_FACTOR).min(self.t_start * 0.3);
                self.stall_count = 0;
                self.reheat_count += 1;
            } else {
                // Slow cooling
                self.temperature *= self.base_cooling_rate.powf(0.5);
            }
        } else if improvement_rate > stall_threshold * 3.0 {
            // Good progress - cool faster
            self.temperature *= self.base_cooling_rate.powf(1.5);
            self.stall_count = 0;
        } else {
            // Normal
            self.temperature *= self.base_cooling_rate;
            self.stall_count = self.stall_count.saturating_sub(1);
        }

        self.temperature = self.temperature.max(self.t_end);
        self.temperature
    }

    fn reset(&mut self, func: TestFunction) {
        self.t_start = func.t_start();
        self.t_end = func.t_end();
        self.temperature = self.t_start;
        self.base_cooling_rate = (self.t_end / self.t_start).powf(1.0 / TOTAL_STEPS as f32);
        self.convergence_threshold = func.convergence_threshold();
        self.energy_history.clear();
        self.stall_count = 0;
        self.reheat_count = 0;
    }
}

struct RunState {
    system: ThermodynamicSystem,
    particles: Vec<ThermodynamicParticle>,
    min_energy: f32,
    mean_energy: f32,
    best_pos: [f32; 2],
    energy_history: Vec<f32>,
    temp_history: Vec<f32>,
    temperature: f32,
}

impl RunState {
    fn new(func: TestFunction) -> Self {
        let t_start = func.t_start();
        let mut system = ThermodynamicSystem::with_loss_function(
            PARTICLE_COUNT, DIM, t_start, func.loss_function()
        );
        system.set_repulsion_samples(64);
        let particles = system.read_particles();

        Self {
            system,
            particles,
            min_energy: f32::MAX,
            mean_energy: 0.0,
            best_pos: [0.0, 0.0],
            energy_history: Vec::new(),
            temp_history: Vec::new(),
            temperature: t_start,
        }
    }

    fn step(&mut self, temperature: f32, func: TestFunction) {
        self.temperature = temperature;
        self.system.set_temperature(temperature);

        let t_start = func.t_start();
        let repulsion = if temperature > t_start * 0.2 { 64 }
                       else if temperature > t_start * 0.02 { 32 }
                       else { 0 };
        self.system.set_repulsion_samples(repulsion);

        // Adaptive dt based on function and temperature
        let dt = match func {
            TestFunction::Rastrigin => {
                if temperature > 0.1 { 0.01 }
                else if temperature > 0.01 { 0.005 }
                else { 0.002 }
            }
            TestFunction::Schwefel => {
                if temperature > 100.0 { 1.0 }
                else if temperature > 10.0 { 0.5 }
                else if temperature > 1.0 { 0.2 }
                else { 0.1 }
            }
        };
        self.system.set_dt(dt);

        self.system.step();
        self.particles = self.system.read_particles();

        let mut sum = 0.0;
        let mut count = 0;
        for p in &self.particles {
            if !p.energy.is_nan() {
                sum += p.energy;
                count += 1;
                if p.energy < self.min_energy {
                    self.min_energy = p.energy;
                    self.best_pos = [p.pos[0], p.pos[1]];
                }
            }
        }
        self.mean_energy = if count > 0 { sum / count as f32 } else { 0.0 };
    }

    fn record_history(&mut self) {
        self.energy_history.push(self.min_energy);
        self.temp_history.push(self.temperature);
    }

    fn reset(&mut self, func: TestFunction) {
        let t_start = func.t_start();
        self.system = ThermodynamicSystem::with_loss_function(
            PARTICLE_COUNT, DIM, t_start, func.loss_function()
        );
        self.system.set_repulsion_samples(64);
        self.particles = self.system.read_particles();
        self.min_energy = f32::MAX;
        self.mean_energy = 0.0;
        self.best_pos = [0.0, 0.0];
        self.energy_history.clear();
        self.temp_history.clear();
        self.temperature = t_start;
    }
}

struct App {
    func: TestFunction,
    fixed: RunState,
    adaptive: RunState,
    adaptive_scheduler: AdaptiveScheduler,
    cache_fixed: canvas::Cache,
    cache_adaptive: canvas::Cache,
    step_count: u32,
    paused: bool,
    speed: u32,
}

impl App {
    fn new() -> Self {
        let func = TestFunction::Rastrigin;
        Self {
            func,
            fixed: RunState::new(func),
            adaptive: RunState::new(func),
            adaptive_scheduler: AdaptiveScheduler::new(func, TOTAL_STEPS),
            cache_fixed: canvas::Cache::new(),
            cache_adaptive: canvas::Cache::new(),
            step_count: 0,
            paused: false,
            speed: 5,
        }
    }

    fn reset(&mut self) {
        self.fixed.reset(self.func);
        self.adaptive.reset(self.func);
        self.adaptive_scheduler.reset(self.func);
        self.step_count = 0;
        self.cache_fixed.clear();
        self.cache_adaptive.clear();
    }

    fn switch_function(&mut self) {
        self.func = match self.func {
            TestFunction::Rastrigin => TestFunction::Schwefel,
            TestFunction::Schwefel => TestFunction::Rastrigin,
        };
        self.reset();
    }

    fn fixed_temperature(&self) -> f32 {
        let progress = (self.step_count as f32 / TOTAL_STEPS as f32).min(1.0);
        let t_start = self.func.t_start();
        let t_end = self.func.t_end();
        t_start * (t_end / t_start).powf(progress)
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
            if app.paused || app.step_count >= TOTAL_STEPS {
                return;
            }

            for _ in 0..app.speed {
                if app.step_count >= TOTAL_STEPS {
                    break;
                }

                let fixed_temp = app.fixed_temperature();
                app.fixed.step(fixed_temp, app.func);

                let adaptive_temp = app.adaptive_scheduler.update(app.adaptive.min_energy);
                app.adaptive.step(adaptive_temp, app.func);

                app.step_count += 1;

                if app.step_count % 10 == 0 {
                    app.fixed.record_history();
                    app.adaptive.record_history();
                }
            }

            app.cache_fixed.clear();
            app.cache_adaptive.clear();
        }
        Message::Event(Event::Keyboard(keyboard::Event::KeyPressed { key, .. })) => {
            match key {
                keyboard::Key::Named(keyboard::key::Named::Space) => app.paused = !app.paused,
                keyboard::Key::Character(c) => match c.as_str() {
                    "r" => app.reset(),
                    "s" | "S" => app.switch_function(),
                    "+" | "=" => app.speed = (app.speed * 2).min(50),
                    "-" => app.speed = (app.speed / 2).max(1),
                    _ => {}
                }
                _ => {}
            }
        }
        Message::Event(_) => {}
    }
}

fn view(app: &App) -> Element<'_, Message> {
    let progress = (app.step_count as f32 / TOTAL_STEPS as f32 * 100.0).min(100.0);
    let status = if app.paused { "PAUSED" }
                 else if app.step_count >= TOTAL_STEPS { "DONE" }
                 else { "RUNNING" };

    let (gx, gy) = app.func.global_opt();
    let info = format!(
        "{} | Step: {}/{} ({:.0}%) | {} | Speed: {}x | Reheats: {} | Global: ({:.0},{:.0})",
        app.func.name(), app.step_count, TOTAL_STEPS, progress, status, app.speed,
        app.adaptive_scheduler.reheat_count, gx, gy
    );

    let fixed_stats = format!(
        "FIXED: T={:.4} | Min={:.2} | Best=({:.1},{:.1})",
        app.fixed.temperature, app.fixed.min_energy,
        app.fixed.best_pos[0], app.fixed.best_pos[1]
    );

    let adaptive_stats = format!(
        "ADAPTIVE: T={:.4} | Min={:.2} | Best=({:.1},{:.1})",
        app.adaptive.temperature, app.adaptive.min_energy,
        app.adaptive.best_pos[0], app.adaptive.best_pos[1]
    );

    let winner = if app.step_count >= TOTAL_STEPS {
        let diff = (app.fixed.min_energy - app.adaptive.min_energy).abs();
        if app.adaptive.min_energy < app.fixed.min_energy - 0.001 {
            format!("ADAPTIVE wins by {:.4}", diff)
        } else if app.fixed.min_energy < app.adaptive.min_energy - 0.001 {
            format!("FIXED wins by {:.4}", diff)
        } else {
            "TIE (within 0.001)".to_string()
        }
    } else {
        "Space=Pause  R=Reset  S=Switch function  +/-=Speed".to_string()
    };

    Column::new()
        .push(Text::new(info).size(16))
        .push(
            Row::new()
                .push(Text::new(fixed_stats).size(14))
                .push(Text::new("  |  ").size(14))
                .push(Text::new(adaptive_stats).size(14))
        )
        .push(Text::new(winner).size(14))
        .push(
            Row::new()
                .push(
                    container(
                        canvas(FixedView { app })
                            .width(Length::Fill)
                            .height(Length::Fill)
                    )
                    .width(Length::FillPortion(1))
                    .height(Length::Fill)
                )
                .push(
                    container(
                        canvas(AdaptiveView { app })
                            .width(Length::Fill)
                            .height(Length::Fill)
                    )
                    .width(Length::FillPortion(1))
                    .height(Length::Fill)
                )
        )
        .into()
}

fn subscription(_app: &App) -> Subscription<Message> {
    Subscription::batch([
        iced::time::every(std::time::Duration::from_millis(16)).map(|_| Message::Tick),
        event::listen().map(Message::Event),
    ])
}

struct FixedView<'a> {
    app: &'a App,
}

struct AdaptiveView<'a> {
    app: &'a App,
}

fn draw_panel(
    frame: &mut canvas::Frame,
    size: iced::Size,
    particles: &[ThermodynamicParticle],
    best_pos: [f32; 2],
    temperature: f32,
    energy_history: &[f32],
    temp_history: &[f32],
    func: TestFunction,
    is_adaptive: bool,
) {
    let (domain_min, domain_max) = func.domain();
    let domain_size = domain_max - domain_min;
    let t_start = func.t_start();
    let t_end = func.t_end();

    // Heatmap
    let resolution = 50;
    let max_loss = match func {
        TestFunction::Rastrigin => 80.0,
        TestFunction::Schwefel => 1700.0,
    };

    for i in 0..resolution {
        for j in 0..resolution {
            let x = domain_min + (i as f32 / resolution as f32) * domain_size;
            let y = domain_min + (j as f32 / resolution as f32) * domain_size;
            let loss = match func {
                TestFunction::Rastrigin => rastrigin(x, y),
                TestFunction::Schwefel => schwefel(x, y),
            };

            let t = (loss / max_loss).clamp(0.0, 1.0);
            let color = if t < 0.05 {
                Color::from_rgb(0.0, 0.6 - t * 4.0, 0.2)
            } else if t < 0.2 {
                Color::from_rgb(0.0, 0.1, 0.3 + t)
            } else {
                Color::from_rgb(t * 0.4, 0.0, 0.2)
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

    // Global optimum marker
    let (gx, gy) = func.global_opt();
    let ox = ((gx - domain_min) / domain_size) * size.width;
    let oy = ((gy - domain_min) / domain_size) * size.height;
    let target = canvas::Path::circle(Point::new(ox, oy), 10.0);
    frame.stroke(&target, canvas::Stroke::default().with_color(Color::from_rgb(0.0, 1.0, 0.0)).with_width(2.0));

    // Particles
    for p in particles {
        if p.pos[0].is_nan() || p.pos[1].is_nan() { continue; }
        let x = ((p.pos[0] - domain_min) / domain_size) * size.width;
        let y = ((p.pos[1] - domain_min) / domain_size) * size.height;

        let t = (p.energy / (max_loss * 0.5)).clamp(0.0, 1.0);
        let color = if is_adaptive {
            Color::from_rgb(1.0 - t * 0.3, 1.0 - t * 0.5, 0.5)
        } else {
            Color::from_rgb(0.5, 1.0 - t * 0.5, 1.0 - t * 0.3)
        };

        let circle = canvas::Path::circle(Point::new(x, y), 2.0);
        frame.fill(&circle, color);
    }

    // Best position
    let bx = ((best_pos[0] - domain_min) / domain_size) * size.width;
    let by = ((best_pos[1] - domain_min) / domain_size) * size.height;
    let best = canvas::Path::circle(Point::new(bx, by), 6.0);
    frame.stroke(&best, canvas::Stroke::default().with_color(Color::from_rgb(1.0, 1.0, 0.0)).with_width(2.0));

    // Temperature bar
    let bar_width = 15.0;
    let bar_height = size.height - 50.0;
    let bar_x = 10.0;
    let bar_y = 25.0;

    let bg = canvas::Path::rectangle(Point::new(bar_x, bar_y), iced::Size::new(bar_width, bar_height));
    frame.fill(&bg, Color::from_rgba(0.0, 0.0, 0.0, 0.5));

    let log_t = (temperature.ln() - t_end.ln()) / (t_start.ln() - t_end.ln());
    let fill_height = log_t.clamp(0.0, 1.0) * bar_height;
    let temp_fill = canvas::Path::rectangle(
        Point::new(bar_x, bar_y + bar_height - fill_height),
        iced::Size::new(bar_width, fill_height),
    );
    let temp_color = Color::from_rgb(log_t.clamp(0.0, 1.0), 0.2, 1.0 - log_t.clamp(0.0, 1.0));
    frame.fill(&temp_fill, temp_color);

    // Energy + temperature history
    if !energy_history.is_empty() {
        let graph_width = 150.0;
        let graph_height = 60.0;
        let graph_x = size.width - graph_width - 10.0;
        let graph_y = size.height - graph_height - 10.0;

        let graph_bg = canvas::Path::rectangle(Point::new(graph_x, graph_y), iced::Size::new(graph_width, graph_height));
        frame.fill(&graph_bg, Color::from_rgba(0.0, 0.0, 0.0, 0.7));

        let len = energy_history.len();
        let max_e = energy_history.iter().cloned().fold(1.0_f32, f32::max);
        let min_e = 0.001_f32;

        // Energy curve (cyan)
        for i in 1..len {
            let x1 = graph_x + ((i - 1) as f32 / len as f32) * graph_width;
            let x2 = graph_x + (i as f32 / len as f32) * graph_width;

            let e1 = energy_history[i-1].max(min_e);
            let e2 = energy_history[i].max(min_e);

            let y1 = graph_y + graph_height - ((e1.ln() - min_e.ln()) / (max_e.ln() - min_e.ln())).clamp(0.0, 1.0) * graph_height;
            let y2 = graph_y + graph_height - ((e2.ln() - min_e.ln()) / (max_e.ln() - min_e.ln())).clamp(0.0, 1.0) * graph_height;

            if !y1.is_nan() && !y2.is_nan() {
                let line = canvas::Path::line(Point::new(x1, y1), Point::new(x2, y2));
                frame.stroke(&line, canvas::Stroke::default().with_color(Color::from_rgb(0.0, 1.0, 1.0)).with_width(1.5));
            }
        }

        // Temperature curve (orange)
        if !temp_history.is_empty() {
            for i in 1..temp_history.len().min(len) {
                let x1 = graph_x + ((i - 1) as f32 / len as f32) * graph_width;
                let x2 = graph_x + (i as f32 / len as f32) * graph_width;

                let t1 = (temp_history[i-1].ln() - t_end.ln()) / (t_start.ln() - t_end.ln());
                let t2 = (temp_history[i].ln() - t_end.ln()) / (t_start.ln() - t_end.ln());

                let y1 = graph_y + graph_height - t1.clamp(0.0, 1.0) * graph_height;
                let y2 = graph_y + graph_height - t2.clamp(0.0, 1.0) * graph_height;

                let line = canvas::Path::line(Point::new(x1, y1), Point::new(x2, y2));
                frame.stroke(&line, canvas::Stroke::default().with_color(Color::from_rgb(1.0, 0.5, 0.0)).with_width(1.0));
            }
        }
    }
}

impl canvas::Program<Message> for FixedView<'_> {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let geo = self.app.cache_fixed.draw(renderer, bounds.size(), |frame| {
            draw_panel(
                frame,
                bounds.size(),
                &self.app.fixed.particles,
                self.app.fixed.best_pos,
                self.app.fixed.temperature,
                &self.app.fixed.energy_history,
                &self.app.fixed.temp_history,
                self.app.func,
                false,
            );
        });
        vec![geo]
    }
}

impl canvas::Program<Message> for AdaptiveView<'_> {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let geo = self.app.cache_adaptive.draw(renderer, bounds.size(), |frame| {
            draw_panel(
                frame,
                bounds.size(),
                &self.app.adaptive.particles,
                self.app.adaptive.best_pos,
                self.app.adaptive.temperature,
                &self.app.adaptive.energy_history,
                &self.app.adaptive.temp_history,
                self.app.func,
                true,
            );
        });
        vec![geo]
    }
}
