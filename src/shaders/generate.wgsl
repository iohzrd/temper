// Random value generation shader for nbody-entropy
// Reads particle positions updated by nbody.wgsl
// Uses u32 operations since WGSL has limited u64 support

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    is_attractor: u32,
    _pad: vec2<f32>,
}

struct GenerateUniforms {
    particle_count: u32,
    output_count: u32,
    state_lo: u32,
    state_hi: u32,
    time: f32,           // Monotonic time for animation
    _pad: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: GenerateUniforms;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

// Shared memory cache for animated positions (cache all particles)
const CACHE_SIZE: u32 = 64u;
var<workgroup> position_cache: array<vec2<f32>, 64>;

const VALUES_PER_THREAD: u32 = 4u;
const TWO_PI: f32 = 6.28318530718;

fn xorshift32(s: u32) -> u32 {
    var x = s;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    return x;
}

// Fast hash using integer operations (no sin)
fn hash_u32(n: u32) -> f32 {
    var x = n;
    x ^= x >> 16u;
    x *= 0x85ebca6bu;
    x ^= x >> 13u;
    x *= 0xc2b2ae35u;
    x ^= x >> 16u;
    return f32(x) / 4294967295.0;
}

// Fast sine approximation using parabola (Bhaskara-style)
// Input: x in radians, output: approximate sin(x)
fn fast_sin(x: f32) -> f32 {
    // Normalize to [-PI, PI]
    var t = x - floor(x / TWO_PI) * TWO_PI;
    if t > 3.14159265 { t -= TWO_PI; }
    // Parabolic approximation: 4/pi * x - 4/pi^2 * x * |x|
    let b = 1.27323954;  // 4/pi
    let c = 0.40528473;  // 4/pi^2
    return b * t - c * t * abs(t);
}

fn fast_cos(x: f32) -> f32 {
    return fast_sin(x + 1.57079632); // cos(x) = sin(x + pi/2)
}

// Get particle position - STATIC for baseline benchmark
fn get_animated_position(idx: u32) -> vec2<f32> {
    return particles[idx].pos;
}

fn get_position(idx: u32) -> vec2<f32> {
    if idx < CACHE_SIZE {
        return position_cache[idx];
    }
    return get_animated_position(idx);
}

fn generate_value(base_lo: u32, base_hi: u32, offset: u32) -> vec2<u32> {
    var lo = base_lo + offset * 2654435761u;
    var hi = base_hi ^ (offset * 1597334677u);

    lo = xorshift32(lo);
    hi = xorshift32(hi ^ lo);

    let mask = uniforms.particle_count - 1u;
    let i = (lo >> 3u) & mask;
    let j = (lo >> 11u) & mask;
    let k = (hi >> 3u) & mask;

    // Get animated positions
    let pi = get_position(i);
    let pj = get_position(j);
    let pk = get_position(k);

    // Mix particle positions (now animated!)
    lo ^= bitcast<u32>(pi.x * 1e8);
    hi ^= bitcast<u32>(pi.y * 1e8);
    lo = lo * 2654435761u;

    lo ^= bitcast<u32>(pj.x * 1e8);
    hi ^= bitcast<u32>(pj.y * 1e8);
    hi = hi * 1597334677u;

    // Mix third particle
    lo ^= bitcast<u32>(pk.x * 1e8);
    hi ^= bitcast<u32>(pk.y * 1e8);

    // Final mixing
    lo = xorshift32(lo);
    hi = xorshift32(hi ^ lo);
    lo = lo * 2246822519u;
    hi = hi * 3266489917u;
    lo ^= lo >> 15u;
    hi ^= hi >> 15u;
    lo ^= hi >> 7u;
    hi ^= lo << 11u;

    return vec2<u32>(lo, hi);
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // Cache animated positions in shared memory (computed once per workgroup)
    if lid < CACHE_SIZE && lid < uniforms.particle_count {
        position_cache[lid] = get_animated_position(lid);
    }
    workgroupBarrier();

    let thread_idx = gid.x;
    let base_idx = thread_idx * VALUES_PER_THREAD;
    if base_idx >= uniforms.output_count { return; }

    let base_lo = uniforms.state_lo + thread_idx * 2654435761u;
    let base_hi = uniforms.state_hi ^ (thread_idx * 1597334677u);

    // Feedback chain: each value feeds into the next within this thread
    var chain_lo = base_lo;
    var chain_hi = base_hi;

    for (var v: u32 = 0u; v < VALUES_PER_THREAD; v++) {
        let idx = base_idx + v;
        if idx >= uniforms.output_count { break; }

        let result = generate_value(chain_lo, chain_hi, v);
        output[idx * 2u] = result.x;
        output[idx * 2u + 1u] = result.y;

        // Feedback: mix output into state for next value
        chain_lo ^= result.x;
        chain_hi ^= result.y;
        chain_lo = xorshift32(chain_lo);
        chain_hi = xorshift32(chain_hi);
    }
}
