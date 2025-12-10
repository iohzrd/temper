//! GPU-accelerated N-body entropy using wgpu compute shaders.
//!
//! This module provides a GPU-accelerated version of the N-body entropy generator.
//! Particles move via chaotic N-body gravitational dynamics.
//!
//! Optimizations:
//! - Two-pass compute: N-body physics + entropy generation
//! - Double buffering: overlap GPU work with CPU readback
//! - Large batch size: 131072 values per roundtrip
//! - Multi-value per thread: each thread generates 4 values
//! - Shared memory caching for particle positions

use bytemuck::{Pod, Zeroable};
use rand_core::{RngCore, SeedableRng};
use std::time::Instant;
use wgpu::util::DeviceExt;

/// Simple seed mixer using splitmix64
fn mix_seed(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Number of particles (static positions for entropy source)
pub const PARTICLE_COUNT: usize = 64;

/// Number of values to generate per GPU roundtrip
const VALUES_PER_BATCH: usize = 131072;

/// Values generated per thread
const VALUES_PER_THREAD: u32 = 4;

/// GPU particle representation (matches nbody.wgsl Particle struct)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParticle {
    pos: [f32; 2],     // Current position
    vel: [f32; 2],     // Current velocity
    mass: f32,         // Particle mass
    is_attractor: u32, // 1 = attractor, 0 = follower
    _pad: [f32; 2],    // Padding for alignment
}

/// GPU uniforms for generation
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GenerateUniforms {
    particle_count: u32,
    output_count: u32,
    state_lo: u32,
    state_hi: u32,
    time: f32, // Monotonic time (unused with n-body, kept for compatibility)
    _pad: f32,
    _pad2: f32,
    _pad3: f32,
}

/// GPU uniforms for n-body physics
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NbodyUniforms {
    attractor_count: u32,
    particle_count: u32,
    g: f32,         // Gravitational constant
    softening: f32, // Softening parameter
    damping: f32,   // Velocity damping
    boundary: f32,  // Boundary size for wrapping
    dt: f32,        // Time step
    steps: u32,     // Number of physics steps per dispatch
}

/// Number of attractor particles (heavier, interact with each other)
const ATTRACTOR_COUNT: usize = 8;

/// Double buffer for pipelining
struct DoubleBuffer {
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// GPU-accelerated chaos entropy generator
pub struct GpuNbodyEntropy {
    device: wgpu::Device,
    queue: wgpu::Queue,
    particle_buffer: wgpu::Buffer,
    nbody_uniform_buffer: wgpu::Buffer,
    nbody_pipeline: wgpu::ComputePipeline,
    nbody_bind_group: wgpu::BindGroup,
    generate_uniform_buffer: wgpu::Buffer,
    generate_pipeline: wgpu::ComputePipeline,
    buffers: [DoubleBuffer; 2],
    pending_buffer: Option<usize>,
    state: u64,
    start_time: Instant,
    value_buffer: Vec<u64>,
    value_index: usize,
}

impl std::fmt::Debug for GpuNbodyEntropy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuNbodyEntropy").finish()
    }
}

impl GpuNbodyEntropy {
    /// Create a new GPU entropy generator, seeded from OS entropy (/dev/urandom on Linux)
    pub fn new() -> Self {
        let mut seed = [0u8; 8];
        getrandom::fill(&mut seed).expect("Failed to get entropy from OS");
        Self::from_seed(seed)
    }

    fn submit_work(&mut self, buffer_idx: usize) {
        let time = self.start_time.elapsed().as_secs_f32();
        let gen_uniforms = GenerateUniforms {
            particle_count: PARTICLE_COUNT as u32,
            output_count: VALUES_PER_BATCH as u32,
            state_lo: self.state as u32,
            state_hi: (self.state >> 32) as u32,
            time,
            _pad: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        };
        self.queue.write_buffer(
            &self.generate_uniform_buffer,
            0,
            bytemuck::bytes_of(&gen_uniforms),
        );

        // Update state for next submission
        self.state = self.state.wrapping_add(VALUES_PER_BATCH as u64);
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Pass 1: N-body physics updates particle positions
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nbody"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.nbody_pipeline);
            pass.set_bind_group(0, &self.nbody_bind_group, &[]);
            pass.dispatch_workgroups((PARTICLE_COUNT as u32 + 63) / 64, 1, 1);
        }

        // Pass 2: Generate random values from particle positions
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("generate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.generate_pipeline);
            pass.set_bind_group(0, &self.buffers[buffer_idx].bind_group, &[]);
            let threads = VALUES_PER_BATCH as u32 / VALUES_PER_THREAD;
            pass.dispatch_workgroups((threads + 255) / 256, 1, 1);
        }

        let size = (VALUES_PER_BATCH * 8) as u64;
        encoder.copy_buffer_to_buffer(
            &self.buffers[buffer_idx].output_buffer,
            0,
            &self.buffers[buffer_idx].staging_buffer,
            0,
            size,
        );

        self.queue.submit(Some(encoder.finish()));
    }

    fn read_buffer(&mut self, buffer_idx: usize) {
        let size = (VALUES_PER_BATCH * 8) as u64;
        let slice = self.buffers[buffer_idx].staging_buffer.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().expect("map failed");

        {
            let data = slice.get_mapped_range();
            let values: &[u32] = bytemuck::cast_slice(&data);
            self.value_buffer.clear();
            for i in 0..VALUES_PER_BATCH {
                let lo = values[i * 2] as u64;
                let hi = values[i * 2 + 1] as u64;
                self.value_buffer.push(lo | (hi << 32));
            }
        }
        self.buffers[buffer_idx].staging_buffer.unmap();
        self.value_index = 0;
    }

    fn fill_buffer(&mut self) {
        if let Some(pending) = self.pending_buffer {
            self.read_buffer(pending);
            let next = 1 - pending;
            self.submit_work(next);
            self.pending_buffer = Some(next);
        } else {
            self.submit_work(0);
            self.read_buffer(0);
            self.submit_work(1);
            self.pending_buffer = Some(1);
        }
    }

    #[inline]
    fn generate(&mut self) -> u64 {
        if self.value_index >= self.value_buffer.len() {
            self.fill_buffer();
        }
        let val = self.value_buffer[self.value_index];
        self.value_index += 1;
        val
    }
}

impl Default for GpuNbodyEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl SeedableRng for GpuNbodyEntropy {
    type Seed = [u8; 8];

    fn from_seed(seed: Self::Seed) -> Self {
        let seed_u64 = u64::from_le_bytes(seed);
        let initial_state = mix_seed(seed_u64);

        // Initialize particles for n-body simulation
        let mut state = initial_state;
        let mut particles = Vec::with_capacity(PARTICLE_COUNT);
        let boundary = 1.0f32;

        for i in 0..PARTICLE_COUNT {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            let pos = [
                (state & 0xFFFF) as f32 / 65535.0 * boundary,
                ((state >> 16) & 0xFFFF) as f32 / 65535.0 * boundary,
            ];

            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            // Small random initial velocity
            let vel = [
                ((state & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.01,
                (((state >> 16) & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.01,
            ];

            let is_attractor = if i < ATTRACTOR_COUNT { 1u32 } else { 0u32 };
            let mass = if is_attractor == 1 { 10.0 } else { 1.0 };

            particles.push(GpuParticle {
                pos,
                vel,
                mass,
                is_attractor,
                _pad: [0.0, 0.0],
            });
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("No GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: Default::default(),
            trace: Default::default(),
        }))
        .expect("Failed to create device");

        // Particle buffer needs read_write for n-body updates
        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // N-body uniforms
        let nbody_uniforms = NbodyUniforms {
            attractor_count: ATTRACTOR_COUNT as u32,
            particle_count: PARTICLE_COUNT as u32,
            g: 0.0001,        // Gravitational constant
            softening: 0.001, // Prevent singularities
            damping: 0.999,   // Slight velocity damping
            boundary: 1.0,    // Wrap at boundaries
            dt: 0.016,        // ~60 FPS timestep
            steps: 4,         // Physics steps per dispatch
        };

        let nbody_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&nbody_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // N-body shader and pipeline
        let nbody_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/nbody.wgsl").into()),
        });

        let nbody_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let nbody_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &nbody_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nbody_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let nbody_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&nbody_bind_group_layout],
                push_constant_ranges: &[],
            });

        let nbody_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&nbody_pipeline_layout),
            module: &nbody_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Generate shader and uniforms
        let start_time = Instant::now();
        let gen_uniforms = GenerateUniforms {
            particle_count: PARTICLE_COUNT as u32,
            output_count: VALUES_PER_BATCH as u32,
            state_lo: initial_state as u32,
            state_hi: (initial_state >> 32) as u32,
            time: 0.0,
            _pad: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        };

        let generate_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&gen_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let generate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/generate.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let create_buffer = |_idx: usize| -> DoubleBuffer {
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (VALUES_PER_BATCH * 8) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (VALUES_PER_BATCH * 8) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: generate_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

            DoubleBuffer {
                output_buffer,
                staging_buffer,
                bind_group,
            }
        };

        let buffers = [create_buffer(0), create_buffer(1)];

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let generate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &generate_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            particle_buffer,
            nbody_uniform_buffer,
            nbody_pipeline,
            nbody_bind_group,
            generate_uniform_buffer,
            generate_pipeline,
            buffers,
            pending_buffer: None,
            state: initial_state,
            start_time,
            value_buffer: Vec::with_capacity(VALUES_PER_BATCH),
            value_index: VALUES_PER_BATCH,
        }
    }
}

impl RngCore for GpuNbodyEntropy {
    fn next_u32(&mut self) -> u32 {
        self.generate() as u32
    }
    fn next_u64(&mut self) -> u64 {
        self.generate()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut i = 0;
        while i < dest.len() {
            let val = self.generate();
            let bytes = val.to_le_bytes();
            let remaining = dest.len() - i;
            let to_copy = remaining.min(8);
            dest[i..i + to_copy].copy_from_slice(&bytes[..to_copy]);
            i += to_copy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_deterministic() {
        let mut rng1 = GpuNbodyEntropy::from_seed([1, 2, 3, 4, 5, 6, 7, 8]);
        let mut rng2 = GpuNbodyEntropy::from_seed([1, 2, 3, 4, 5, 6, 7, 8]);
        for _ in 0..10 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_gpu_bit_distribution() {
        let mut rng = GpuNbodyEntropy::from_seed([42, 0, 0, 0, 0, 0, 0, 0]);
        let mut ones = 0u64;
        for _ in 0..1000 {
            ones += rng.next_u64().count_ones() as u64;
        }
        let expected = 1000 * 32;
        let deviation = (ones as f64 - expected as f64).abs() / expected as f64;
        assert!(
            deviation < 0.05,
            "Bit distribution deviation: {}",
            deviation
        );
    }
}
