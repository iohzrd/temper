//! Particle visualization for nbody-entropy using iced
//!
//! Run with: cargo run --release --features viz --bin nbody-viz

use iced::mouse;
use iced::widget::{canvas, container};
use iced::{Color, Element, Length, Point, Rectangle, Renderer, Size, Subscription, Theme};

const PARTICLE_COUNT: usize = 64;
const ATTRACTOR_COUNT: usize = 8;
const MAX_LINE_DISTANCE: f32 = 80.0;

// N-body physics constants
const G: f32 = 0.0001;
const SOFTENING: f32 = 0.001;
const DAMPING: f32 = 0.999;
const BOUNDARY: f32 = 1.0;
const DT: f32 = 0.016;
const SLINGSHOT_RADIUS: f32 = 0.08; // Distance below which slingshot kicks in
const SLINGSHOT_STRENGTH: f32 = 0.002; // Tangential velocity boost strength

fn main() -> iced::Result {
    iced::application(App::default, update, view)
        .subscription(subscription)
        .title("N-Body Entropy - Visualization")
        .run()
}

#[derive(Default)]
struct App {
    particles: ParticleSystem,
}

#[derive(Debug, Clone)]
enum Message {
    Tick,
}

fn update(app: &mut App, message: Message) {
    match message {
        Message::Tick => {
            app.particles.step_physics();
            app.particles.request_redraw();
        }
    }
}

fn view(app: &App) -> Element<'_, Message> {
    container(
        canvas(&app.particles)
            .width(Length::Fill)
            .height(Length::Fill),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .style(|_| container::Style {
        background: Some(iced::Background::Color(Color::from_rgb(0.05, 0.05, 0.1))),
        ..Default::default()
    })
    .into()
}

fn subscription(_app: &App) -> Subscription<Message> {
    iced::time::every(std::time::Duration::from_millis(16)).map(|_| Message::Tick)
}

struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
    mass: f32,
    is_attractor: bool,
}

struct ParticleSystem {
    particles: Vec<Particle>,
    cache: canvas::Cache,
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self::new(42)
    }
}

/// Wrap position to stay within bounds
#[inline]
fn wrap(x: f32) -> f32 {
    let x = x % BOUNDARY;
    if x < 0.0 { x + BOUNDARY } else { x }
}

/// Compute shortest distance with wrapping
#[inline]
fn wrapped_delta(a: f32, b: f32) -> f32 {
    let mut d = b - a;
    let half = BOUNDARY / 2.0;
    if d > half {
        d -= BOUNDARY;
    }
    if d < -half {
        d += BOUNDARY;
    }
    d
}

impl ParticleSystem {
    fn new(seed: u64) -> Self {
        let mut state = seed;
        let mut particles = Vec::with_capacity(PARTICLE_COUNT);

        for i in 0..PARTICLE_COUNT {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            let pos = [
                (state & 0xFFFF) as f32 / 65535.0 * BOUNDARY,
                ((state >> 16) & 0xFFFF) as f32 / 65535.0 * BOUNDARY,
            ];

            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            let vel = [
                ((state & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.01,
                (((state >> 16) & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.01,
            ];

            let is_attractor = i < ATTRACTOR_COUNT;
            let mass = if is_attractor { 10.0 } else { 1.0 };

            particles.push(Particle {
                pos,
                vel,
                mass,
                is_attractor,
            });
        }

        Self {
            particles,
            cache: canvas::Cache::new(),
        }
    }

    fn request_redraw(&self) {
        self.cache.clear();
    }

    /// Step the n-body simulation forward
    fn step_physics(&mut self) {
        // Compute forces on attractors from other attractors
        let mut attractor_forces = [[0.0f32; 2]; ATTRACTOR_COUNT];

        for i in 0..ATTRACTOR_COUNT {
            for j in (i + 1)..ATTRACTOR_COUNT {
                let pi = self.particles[i].pos;
                let pj = self.particles[j].pos;

                let dx = wrapped_delta(pi[0], pj[0]);
                let dy = wrapped_delta(pi[1], pj[1]);

                let dist_sq = dx * dx + dy * dy + SOFTENING;
                let dist = dist_sq.sqrt();

                // Gravitational attraction
                let gravity = G * self.particles[i].mass * self.particles[j].mass / dist_sq;

                let mut fx = gravity * dx / dist;
                let mut fy = gravity * dy / dist;

                // Slingshot effect: tangential velocity boost on close approach
                if dist < SLINGSHOT_RADIUS {
                    let tangent_x = -dy / dist;
                    let tangent_y = dx / dist;
                    let boost = SLINGSHOT_STRENGTH / dist_sq;
                    fx += boost * tangent_x;
                    fy += boost * tangent_y;
                }

                attractor_forces[i][0] += fx;
                attractor_forces[i][1] += fy;
                attractor_forces[j][0] -= fx;
                attractor_forces[j][1] -= fy;
            }
        }

        // Update attractor velocities and positions
        for i in 0..ATTRACTOR_COUNT {
            let p = &mut self.particles[i];
            p.vel[0] += attractor_forces[i][0] * DT / p.mass;
            p.vel[1] += attractor_forces[i][1] * DT / p.mass;
            p.vel[0] *= DAMPING;
            p.vel[1] *= DAMPING;
            p.pos[0] = wrap(p.pos[0] + p.vel[0] * DT);
            p.pos[1] = wrap(p.pos[1] + p.vel[1] * DT);
        }

        // Update followers - influenced by attractors
        for i in ATTRACTOR_COUNT..PARTICLE_COUNT {
            let mut fx = 0.0;
            let mut fy = 0.0;

            // Sample a few attractors for speed
            let sample_count = 4.min(ATTRACTOR_COUNT);
            let step = ATTRACTOR_COUNT / sample_count;
            for s in 0..sample_count {
                let j = s * step;
                let pi = self.particles[i].pos;
                let pj = self.particles[j].pos;

                let dx = wrapped_delta(pi[0], pj[0]);
                let dy = wrapped_delta(pi[1], pj[1]);

                let dist_sq = dx * dx + dy * dy + SOFTENING;
                let dist = dist_sq.sqrt();

                // Gravitational attraction
                let gravity = G * self.particles[j].mass / dist_sq;

                fx += gravity * dx / dist;
                fy += gravity * dy / dist;

                // Slingshot effect: tangential velocity boost on close approach
                if dist < SLINGSHOT_RADIUS {
                    let tangent_x = -dy / dist;
                    let tangent_y = dx / dist;
                    let boost = SLINGSHOT_STRENGTH / dist_sq;
                    fx += boost * tangent_x;
                    fy += boost * tangent_y;
                }
            }

            let p = &mut self.particles[i];
            p.vel[0] += fx * DT;
            p.vel[1] += fy * DT;
            p.vel[0] *= DAMPING;
            p.vel[1] *= DAMPING;
            p.pos[0] = wrap(p.pos[0] + p.vel[0] * DT);
            p.pos[1] = wrap(p.pos[1] + p.vel[1] * DT);
        }
    }

    fn get_position(&self, p: &Particle, bounds: Size) -> Point {
        Point::new(p.pos[0] * bounds.width, p.pos[1] * bounds.height)
    }
}

impl canvas::Program<Message> for ParticleSystem {
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
            // Get all current positions
            let positions: Vec<Point> = self
                .particles
                .iter()
                .map(|p| self.get_position(p, bounds.size()))
                .collect();

            // Draw lines between nearby particles
            for i in 0..positions.len() {
                for j in (i + 1)..positions.len() {
                    let p1 = positions[i];
                    let p2 = positions[j];
                    let dx = p2.x - p1.x;
                    let dy = p2.y - p1.y;
                    let dist = (dx * dx + dy * dy).sqrt();

                    if dist < MAX_LINE_DISTANCE {
                        let alpha = 1.0 - (dist / MAX_LINE_DISTANCE);
                        // Attractor-attractor lines are brighter
                        let is_attractor_pair = i < ATTRACTOR_COUNT && j < ATTRACTOR_COUNT;
                        let color = if is_attractor_pair {
                            Color::from_rgba(1.0, 0.6, 0.3, alpha * 0.5)
                        } else {
                            Color::from_rgba(0.3, 0.6, 1.0, alpha * 0.3)
                        };
                        let line = canvas::Path::line(p1, p2);
                        frame.stroke(
                            &line,
                            canvas::Stroke::default().with_color(color).with_width(1.0),
                        );
                    }
                }
            }

            // Draw particles - attractors are larger and orange
            for (i, pos) in positions.iter().enumerate() {
                let is_attractor = i < ATTRACTOR_COUNT;
                let (radius, color) = if is_attractor {
                    (5.0, Color::from_rgb(1.0, 0.5, 0.2))
                } else {
                    (2.5, Color::from_rgb(0.4, 0.7, 1.0))
                };
                let circle = canvas::Path::circle(*pos, radius);
                frame.fill(&circle, color);
            }
        });

        vec![geometry]
    }
}
