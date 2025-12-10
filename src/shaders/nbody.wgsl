// N-body compute shader for nbody-entropy
// Computes gravitational forces and updates particle positions/velocities

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    is_attractor: u32, // 1 = attractor, 0 = follower
    _pad: vec2<f32>,
}

struct Uniforms {
    attractor_count: u32,
    particle_count: u32,
    g: f32,             // Gravitational constant
    softening: f32,     // Softening parameter
    damping: f32,       // Velocity damping
    boundary: f32,      // Boundary size for wrapping
    dt: f32,            // Time step
    steps: u32,         // Number of physics steps to run
}

// Slingshot constants (tangential velocity boost on close approach)
const SLINGSHOT_RADIUS: f32 = 0.08;
const SLINGSHOT_STRENGTH: f32 = 0.002;

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

// Wrap position to stay within bounds
fn wrap(x: f32, boundary: f32) -> f32 {
    var result = x % boundary;
    if result < 0.0 {
        result = result + boundary;
    }
    return result;
}

// Compute shortest distance with wrapping
fn wrapped_delta(a: f32, b: f32, boundary: f32) -> f32 {
    var d = b - a;
    let half = boundary / 2.0;
    if d > half {
        d = d - boundary;
    }
    if d < -half {
        d = d + boundary;
    }
    return d;
}

// Main compute shader - each invocation handles one particle
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.particle_count {
        return;
    }

    // Run multiple physics steps
    for (var step = 0u; step < uniforms.steps; step = step + 1u) {
        // Synchronize between steps (all threads must complete before next step)
        workgroupBarrier();
        storageBarrier();

        var p = particles[idx];
        var force = vec2<f32>(0.0, 0.0);

        if p.is_attractor == 1u {
            // Attractors interact with all other attractors
            for (var j = 0u; j < uniforms.attractor_count; j = j + 1u) {
                if j == idx {
                    continue;
                }

                let other = particles[j];
                let dx = wrapped_delta(p.pos.x, other.pos.x, uniforms.boundary);
                let dy = wrapped_delta(p.pos.y, other.pos.y, uniforms.boundary);

                let dist_sq = dx * dx + dy * dy + uniforms.softening;
                let dist = sqrt(dist_sq);

                // Gravitational attraction
                let gravity = uniforms.g * p.mass * other.mass / dist_sq;

                force.x = force.x + gravity * dx / dist;
                force.y = force.y + gravity * dy / dist;

                // Slingshot effect: tangential velocity boost on close approach
                if dist < SLINGSHOT_RADIUS {
                    let tangent_x = -dy / dist;
                    let tangent_y = dx / dist;
                    let boost = SLINGSHOT_STRENGTH / dist_sq;
                    force.x = force.x + boost * tangent_x;
                    force.y = force.y + boost * tangent_y;
                }
            }
        } else {
            // Followers are influenced by attractors only (sample 4 for speed)
            let sample_count = min(4u, uniforms.attractor_count);
            let step_size = uniforms.attractor_count / sample_count;

            for (var s = 0u; s < sample_count; s = s + 1u) {
                let j = s * step_size;
                let other = particles[j];

                let dx = wrapped_delta(p.pos.x, other.pos.x, uniforms.boundary);
                let dy = wrapped_delta(p.pos.y, other.pos.y, uniforms.boundary);

                let dist_sq = dx * dx + dy * dy + uniforms.softening;
                let dist = sqrt(dist_sq);

                // Gravitational attraction
                let gravity = uniforms.g * other.mass / dist_sq;

                force.x = force.x + gravity * dx / dist;
                force.y = force.y + gravity * dy / dist;

                // Slingshot effect: tangential velocity boost on close approach
                if dist < SLINGSHOT_RADIUS {
                    let tangent_x = -dy / dist;
                    let tangent_y = dx / dist;
                    let boost = SLINGSHOT_STRENGTH / dist_sq;
                    force.x = force.x + boost * tangent_x;
                    force.y = force.y + boost * tangent_y;
                }
            }
        }

        // Update velocity
        let accel = force / p.mass;
        p.vel = p.vel + accel * uniforms.dt;
        p.vel = p.vel * uniforms.damping;

        // Update position with wrapping
        p.pos.x = wrap(p.pos.x + p.vel.x * uniforms.dt, uniforms.boundary);
        p.pos.y = wrap(p.pos.y + p.vel.y * uniforms.dt, uniforms.boundary);

        particles[idx] = p;
    }
}
