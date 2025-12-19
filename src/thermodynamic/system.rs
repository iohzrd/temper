//! The core ThermodynamicSystem GPU particle simulation
//!
//! Uses Hamiltonian Monte Carlo (HMC) with leapfrog integration.

use super::types::{
    DiversityMetrics, LossFunction, MAX_DIMENSIONS, MAX_PARTICLES, ParticleScalars,
    ThermodynamicMode, ThermodynamicParticle, ThermodynamicStats, Uniforms,
};
use half::f16;
use wgpu::util::DeviceExt;

/// Get random seed from OS entropy via getrandom
fn random_seed() -> u64 {
    let mut buf = [0u8; 8];
    getrandom::fill(&mut buf).expect("Failed to get OS entropy");
    u64::from_le_bytes(buf)
}

/// Unified thermodynamic particle system using HMC
///
/// Uses Hamiltonian Monte Carlo with temperature-controlled exploration:
///
/// - T >> 1.0  : Entropy mode - large momenta, chaotic exploration
/// - T ~ 0.1   : Sampling mode - balanced, efficient posterior sampling
/// - T → 0    : Optimize mode - small momenta, momentum-based gradient descent
pub struct ThermodynamicSystem {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // SoA buffers for optimal GPU coalescing (dimension-major layout)
    positions_buffer: wgpu::Buffer,  // [dim][particle] f16 positions
    velocities_buffer: wgpu::Buffer, // [dim][particle] f16 velocities
    scalars_buffer: wgpu::Buffer,    // [particle] ParticleScalars
    // Proposal buffers for HMC accept/reject
    proposal_pos_buffer: wgpu::Buffer,
    proposal_vel_buffer: wgpu::Buffer,
    proposal_scalars_buffer: wgpu::Buffer,
    // Other buffers
    entropy_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    accept_flags_buffer: wgpu::Buffer,
    // Staging buffers for CPU readback
    positions_staging: wgpu::Buffer,
    velocities_staging: wgpu::Buffer,
    scalars_staging: wgpu::Buffer,
    entropy_staging: wgpu::Buffer,
    accept_flags_staging: wgpu::Buffer,
    // Pipelines
    init_pipeline: wgpu::ComputePipeline,
    refresh_momentum_pipeline: wgpu::ComputePipeline,
    leapfrog_pipeline: wgpu::ComputePipeline,
    accept_reject_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    particle_count: usize,
    dim: usize,
    temperature: f32,
    step: u32,
    loss_fn: LossFunction,
    base_seed: u32,
    #[allow(dead_code)]
    custom_loss_wgsl: Option<String>,
    // HMC parameters
    leapfrog_steps: u32,
    step_size: f32,
    mass: f32,
    // Position domain bounds
    pos_min: f32,
    pos_max: f32,
}

impl ThermodynamicSystem {
    pub fn new(particle_count: usize, dim: usize, temperature: f32) -> Self {
        Self::with_loss_function(particle_count, dim, temperature, LossFunction::default())
    }

    /// Create with a specific loss function
    pub fn with_loss_function(
        particle_count: usize,
        dim: usize,
        temperature: f32,
        loss_fn: LossFunction,
    ) -> Self {
        assert!(particle_count <= MAX_PARTICLES);
        assert!(dim <= MAX_DIMENSIONS);

        let base_seed = random_seed();

        // Initialize SoA data on CPU (dimension-major layout)
        // positions[d * particle_count + idx] = position of particle idx in dimension d
        let pos_vel_size = dim * particle_count;
        let mut positions: Vec<f16> = vec![f16::ZERO; pos_vel_size];
        let velocities: Vec<f16> = vec![f16::ZERO; pos_vel_size];
        let mut scalars: Vec<ParticleScalars> = vec![ParticleScalars::default(); particle_count];

        // Initialize positions with random values
        let mut seed = base_seed;
        for idx in 0..particle_count {
            for d in 0..dim {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let val = -4.0 + (seed & 0xFFFF) as f32 / 65535.0 * 8.0;
                positions[d * particle_count + idx] = f16::from_f32(val);
            }
            scalars[idx].mass = 1.0;
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("No GPU adapter found");

        // Use adapter limits as base, then override what we need
        let mut limits = adapter.limits();
        limits.max_storage_buffer_binding_size = 1536 * 1024 * 1024; // 1.5 GB
        limits.max_buffer_size = 1536 * 1024 * 1024;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::SHADER_F16,
            required_limits: limits,
            ..Default::default()
        }))
        .expect("Failed to create device with f16 support");

        // SoA buffer sizes
        let pos_vel_buffer_size = (pos_vel_size * std::mem::size_of::<f16>()) as u64;
        let scalars_buffer_size = (particle_count * std::mem::size_of::<ParticleScalars>()) as u64;

        // Create SoA data buffers
        let positions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("positions"),
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let velocities_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("velocities"),
            contents: bytemuck::cast_slice(&velocities),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let scalars_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scalars"),
            contents: bytemuck::cast_slice(&scalars),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Create proposal buffers (for HMC accept/reject)
        let proposal_pos_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proposal_pos"),
            size: pos_vel_buffer_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let proposal_vel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proposal_vel"),
            size: pos_vel_buffer_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let proposal_scalars_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proposal_scalars"),
            size: scalars_buffer_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create staging buffers for CPU readback
        let positions_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("positions_staging"),
            size: pos_vel_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let velocities_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("velocities_staging"),
            size: pos_vel_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scalars_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scalars_staging"),
            size: scalars_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let entropy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entropy"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let entropy_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entropy_staging"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let accept_flags_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accept_flags"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let accept_flags_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accept_flags_staging"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // HMC default parameters
        let leapfrog_steps = 10u32;
        let step_size = 0.1 / (dim as f32).sqrt();
        let mass = 1.0f32;

        // Determine position bounds based on loss function
        let (pos_min, pos_max) = match loss_fn {
            LossFunction::Schwefel => (-500.0, 500.0),
            LossFunction::Custom => (0.0, 1.0), // Default for image diffusion
            _ => (-5.0, 5.0),                   // Standard benchmark bounds
        };

        let uniforms = Uniforms {
            particle_count: particle_count as u32,
            dim: dim as u32,
            temperature,
            seed: base_seed as u32,
            mode: ThermodynamicMode::from_temperature(temperature) as u32,
            loss_fn: loss_fn as u32,
            leapfrog_steps,
            step_size,
            mass,
            _padding: 0,
            pos_min,
            pos_max,
            _padding2: [0, 0],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("thermodynamic"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/thermodynamic.wgsl").into()),
        });

        // Bind group layout: 9 bindings for SoA buffers
        // 0: positions, 1: uniforms, 2: velocities, 3: scalars, 4: entropy_output
        // 5: proposal_pos, 6: proposal_vel, 7: proposal_scalars, 8: accept_flags
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("thermodynamic_layout"),
            entries: &(0..9u32)
                .map(|i| wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if i == 1 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage { read_only: false }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });

        // Bind group with 9 SoA buffers
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thermodynamic_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: velocities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scalars_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: entropy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: proposal_pos_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: proposal_vel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: proposal_scalars_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: accept_flags_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("thermodynamic_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Initialization pipeline (GPU-accelerated particle init)
        let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("init_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("initialize_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        // HMC pipelines
        let refresh_momentum_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("refresh_momentum_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("refresh_momentum"),
                compilation_options: Default::default(),
                cache: None,
            });

        let leapfrog_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("leapfrog_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("leapfrog_step"),
            compilation_options: Default::default(),
            cache: None,
        });

        let accept_reject_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("accept_reject_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("accept_reject"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            device,
            queue,
            positions_buffer,
            velocities_buffer,
            scalars_buffer,
            proposal_pos_buffer,
            proposal_vel_buffer,
            proposal_scalars_buffer,
            entropy_buffer,
            uniform_buffer,
            accept_flags_buffer,
            positions_staging,
            velocities_staging,
            scalars_staging,
            entropy_staging,
            accept_flags_staging,
            init_pipeline,
            refresh_momentum_pipeline,
            leapfrog_pipeline,
            accept_reject_pipeline,
            bind_group,
            particle_count,
            dim,
            temperature,
            step: 0,
            loss_fn,
            base_seed: base_seed as u32,
            custom_loss_wgsl: None,
            leapfrog_steps,
            step_size,
            mass,
            pos_min,
            pos_max,
        }
    }

    /// Create with a custom expression-based loss function
    pub fn with_expr(
        particle_count: usize,
        dim: usize,
        temperature: f32,
        expr: crate::expr::Expr,
    ) -> Self {
        Self::with_expr_options(particle_count, dim, temperature, expr, true)
    }

    /// Create with a custom expression and control gradient type
    pub fn with_expr_options(
        particle_count: usize,
        dim: usize,
        temperature: f32,
        expr: crate::expr::Expr,
        analytical_gradients: bool,
    ) -> Self {
        assert!(particle_count <= MAX_PARTICLES);
        assert!(dim <= MAX_DIMENSIONS);

        let base_seed = random_seed();

        // Initialize SoA data on CPU (dimension-major layout)
        let pos_vel_size = dim * particle_count;
        let mut positions: Vec<f16> = vec![f16::ZERO; pos_vel_size];
        let velocities: Vec<f16> = vec![f16::ZERO; pos_vel_size];
        let mut scalars: Vec<ParticleScalars> = vec![ParticleScalars::default(); particle_count];

        let mut seed = base_seed;
        for idx in 0..particle_count {
            for d in 0..dim {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let val = -4.0 + (seed & 0xFFFF) as f32 / 65535.0 * 8.0;
                positions[d * particle_count + idx] = f16::from_f32(val);
            }
            scalars[idx].mass = 1.0;
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("No GPU adapter found");

        let mut limits = adapter.limits();
        limits.max_storage_buffer_binding_size = 1536 * 1024 * 1024;
        limits.max_buffer_size = 1536 * 1024 * 1024;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::SHADER_F16,
            required_limits: limits,
            ..Default::default()
        }))
        .expect("Failed to create device with f16 support");

        // SoA buffer sizes
        let pos_vel_buffer_size = (pos_vel_size * std::mem::size_of::<f16>()) as u64;
        let scalars_buffer_size = (particle_count * std::mem::size_of::<ParticleScalars>()) as u64;

        // Create SoA data buffers
        let positions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("positions"),
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let velocities_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("velocities"),
            contents: bytemuck::cast_slice(&velocities),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let scalars_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scalars"),
            contents: bytemuck::cast_slice(&scalars),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Create proposal buffers
        let proposal_pos_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proposal_pos"),
            size: pos_vel_buffer_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let proposal_vel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proposal_vel"),
            size: pos_vel_buffer_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let proposal_scalars_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proposal_scalars"),
            size: scalars_buffer_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create staging buffers
        let positions_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("positions_staging"),
            size: pos_vel_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let velocities_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("velocities_staging"),
            size: pos_vel_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scalars_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scalars_staging"),
            size: scalars_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let entropy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entropy"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let entropy_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entropy_staging"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let accept_flags_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accept_flags"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let accept_flags_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accept_flags_staging"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // HMC default parameters
        let leapfrog_steps = 10u32;
        let step_size = 0.1 / (dim as f32).sqrt();
        let mass = 1.0f32;
        let pos_min = 0.0f32;
        let pos_max = 1.0f32;

        let uniforms = Uniforms {
            particle_count: particle_count as u32,
            dim: dim as u32,
            temperature,
            seed: base_seed as u32,
            mode: ThermodynamicMode::from_temperature(temperature) as u32,
            loss_fn: LossFunction::Custom as u32,
            leapfrog_steps,
            step_size,
            mass,
            _padding: 0,
            pos_min,
            pos_max,
            _padding2: [0, 0],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let custom_wgsl = expr.to_wgsl_with_options(analytical_gradients);
        let base_shader = include_str!("../shaders/thermodynamic.wgsl");

        let stub_marker = "// Stub functions for custom expressions";
        let stub_end = "// 2D neural net:";

        let full_shader = if let Some(stub_start) = base_shader.find(stub_marker) {
            if let Some(stub_end_pos) = base_shader[stub_start..].find(stub_end) {
                let before = &base_shader[..stub_start];
                let after = &base_shader[stub_start + stub_end_pos..];
                format!("{}{}\n\n{}", before, custom_wgsl, after)
            } else {
                format!("{}\n\n{}", custom_wgsl, base_shader)
            }
        } else {
            format!("{}\n\n{}", custom_wgsl, base_shader)
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("thermodynamic_custom"),
            source: wgpu::ShaderSource::Wgsl(full_shader.clone().into()),
        });

        // Bind group layout: 9 bindings for SoA buffers
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("thermodynamic_layout"),
            entries: &(0..9u32)
                .map(|i| wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if i == 1 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage { read_only: false }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thermodynamic_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: velocities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scalars_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: entropy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: proposal_pos_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: proposal_vel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: proposal_scalars_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: accept_flags_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("thermodynamic_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("init_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("initialize_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        let refresh_momentum_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("refresh_momentum_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("refresh_momentum"),
                compilation_options: Default::default(),
                cache: None,
            });

        let leapfrog_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("leapfrog_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("leapfrog_step"),
            compilation_options: Default::default(),
            cache: None,
        });

        let accept_reject_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("accept_reject_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("accept_reject"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            device,
            queue,
            positions_buffer,
            velocities_buffer,
            scalars_buffer,
            proposal_pos_buffer,
            proposal_vel_buffer,
            proposal_scalars_buffer,
            entropy_buffer,
            uniform_buffer,
            accept_flags_buffer,
            positions_staging,
            velocities_staging,
            scalars_staging,
            entropy_staging,
            accept_flags_staging,
            init_pipeline,
            refresh_momentum_pipeline,
            leapfrog_pipeline,
            accept_reject_pipeline,
            bind_group,
            particle_count,
            dim,
            temperature,
            step: 0,
            loss_fn: LossFunction::Custom,
            base_seed: base_seed as u32,
            custom_loss_wgsl: Some(full_shader),
            leapfrog_steps,
            step_size,
            mass,
            pos_min,
            pos_max,
        }
    }

    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }

    /// Set the loss function
    pub fn set_loss_function(&mut self, loss_fn: LossFunction) {
        self.loss_fn = loss_fn;
    }

    /// Set position domain bounds
    ///
    /// Different loss functions need different bounds:
    /// - Image diffusion: [0.0, 1.0]
    /// - Schwefel: [-500.0, 500.0]
    /// - Standard benchmarks: [-5.0, 5.0]
    pub fn set_position_bounds(&mut self, min: f32, max: f32) {
        self.pos_min = min;
        self.pos_max = max;
    }

    /// Get current position bounds
    pub fn position_bounds(&self) -> (f32, f32) {
        (self.pos_min, self.pos_max)
    }

    /// Re-initialize all particles on GPU with random positions
    ///
    /// Positions are uniformly distributed in [pos_min, pos_max].
    /// Much faster than CPU initialization for large particle counts.
    /// Useful for restarting optimization or exploring different random initializations.
    pub fn initialize_gpu(&mut self) {
        // Update seed for fresh initialization
        self.base_seed = self.base_seed.wrapping_add(12345);

        let uniforms = Uniforms {
            particle_count: self.particle_count as u32,
            dim: self.dim as u32,
            temperature: self.temperature,
            seed: self.base_seed,
            mode: self.mode() as u32,
            loss_fn: self.loss_fn as u32,
            leapfrog_steps: self.leapfrog_steps,
            step_size: self.step_size,
            mass: self.mass,
            _padding: 0,
            pos_min: self.pos_min,
            pos_max: self.pos_max,
            _padding2: [0, 0],
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("init_encoder"),
            });

        let workgroups = (self.particle_count as u32 + 255) / 256;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("init_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.init_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.step = 0;
    }

    /// Get current loss function
    pub fn loss_function(&self) -> LossFunction {
        self.loss_fn
    }

    /// Get current operating mode
    pub fn mode(&self) -> ThermodynamicMode {
        ThermodynamicMode::from_temperature(self.temperature)
    }

    /// Run one HMC iteration: refresh momentum + L leapfrog steps + accept/reject
    ///
    /// Optimized: All L leapfrog steps run in a single GPU dispatch with vectorized gradients.
    /// This gives O(L*D) complexity instead of O(L*D²) for functions like Ackley.
    pub fn step(&mut self) {
        self.step += 1;

        let uniforms = Uniforms {
            particle_count: self.particle_count as u32,
            dim: self.dim as u32,
            temperature: self.temperature,
            seed: self
                .base_seed
                .wrapping_add(self.step.wrapping_mul(2654435769)),
            mode: self.mode() as u32,
            loss_fn: self.loss_fn as u32,
            leapfrog_steps: self.leapfrog_steps,
            step_size: self.step_size,
            mass: self.mass,
            _padding: 0,
            pos_min: self.pos_min,
            pos_max: self.pos_max,
            _padding2: [0, 0],
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hmc_encoder"),
            });

        let workgroups = (self.particle_count as u32 + 255) / 256;

        // Pass 1: Refresh momentum from Maxwell-Boltzmann distribution
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("refresh_momentum_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.refresh_momentum_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 2: Batched leapfrog integration (all L steps in ONE dispatch)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("leapfrog_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.leapfrog_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 3: Accept/reject based on Hamiltonian change
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("accept_reject_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.accept_reject_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Step without Metropolis accept/reject - pure leapfrog integration
    ///
    /// This is better for optimization tasks where we want gradient-following
    /// behavior without the sampling guarantees of HMC.
    ///
    /// Optimized: All L leapfrog steps run in a single GPU dispatch with vectorized gradients.
    pub fn step_leapfrog_only(&mut self) {
        self.step += 1;

        let uniforms = Uniforms {
            particle_count: self.particle_count as u32,
            dim: self.dim as u32,
            temperature: self.temperature,
            seed: self
                .base_seed
                .wrapping_add(self.step.wrapping_mul(2654435769)),
            mode: self.mode() as u32,
            loss_fn: self.loss_fn as u32,
            leapfrog_steps: self.leapfrog_steps,
            step_size: self.step_size,
            mass: self.mass,
            _padding: 0,
            pos_min: self.pos_min,
            pos_max: self.pos_max,
            _padding2: [0, 0],
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("leapfrog_encoder"),
            });

        let workgroups = (self.particle_count as u32 + 255) / 256;

        // Pass 1: Refresh momentum (with very low T = near-zero momentum = gradient descent)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("refresh_momentum_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.refresh_momentum_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 2: Batched leapfrog integration (all L steps in ONE dispatch)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("leapfrog_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.leapfrog_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Skip Pass 3 (accept/reject) - always keep the leapfrog result

        self.queue.submit(Some(encoder.finish()));
    }

    /// Read particles back from GPU (reconstructing from SoA buffers)
    pub fn read_particles(&self) -> Vec<ThermodynamicParticle> {
        let pos_vel_size = (self.dim * self.particle_count * std::mem::size_of::<f16>()) as u64;
        let scalars_size = (self.particle_count * std::mem::size_of::<ParticleScalars>()) as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.positions_buffer,
            0,
            &self.positions_staging,
            0,
            pos_vel_size,
        );
        encoder.copy_buffer_to_buffer(
            &self.velocities_buffer,
            0,
            &self.velocities_staging,
            0,
            pos_vel_size,
        );
        encoder.copy_buffer_to_buffer(
            &self.scalars_buffer,
            0,
            &self.scalars_staging,
            0,
            scalars_size,
        );
        self.queue.submit(Some(encoder.finish()));

        // Map all three buffers
        let pos_slice = self.positions_staging.slice(..pos_vel_size);
        let vel_slice = self.velocities_staging.slice(..pos_vel_size);
        let scalars_slice = self.scalars_staging.slice(..scalars_size);

        let (tx1, rx1) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        let (tx2, rx2) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        let (tx3, rx3) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();

        pos_slice.map_async(wgpu::MapMode::Read, move |r| {
            tx1.send(r).unwrap();
        });
        vel_slice.map_async(wgpu::MapMode::Read, move |r| {
            tx2.send(r).unwrap();
        });
        scalars_slice.map_async(wgpu::MapMode::Read, move |r| {
            tx3.send(r).unwrap();
        });

        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        rx1.recv().unwrap().expect("positions map failed");
        rx2.recv().unwrap().expect("velocities map failed");
        rx3.recv().unwrap().expect("scalars map failed");

        // Read data
        let pos_data = pos_slice.get_mapped_range();
        let vel_data = vel_slice.get_mapped_range();
        let scalars_data = scalars_slice.get_mapped_range();

        let positions: &[f16] = bytemuck::cast_slice(&pos_data);
        let velocities: &[f16] = bytemuck::cast_slice(&vel_data);
        let scalars: &[ParticleScalars] = bytemuck::cast_slice(&scalars_data);

        // Reconstruct particles from SoA (dimension-major) to AoS
        let mut particles = vec![ThermodynamicParticle::default(); self.particle_count];
        for idx in 0..self.particle_count {
            for d in 0..self.dim {
                particles[idx].pos[d] = positions[d * self.particle_count + idx];
                particles[idx].vel[d] = velocities[d * self.particle_count + idx];
            }
            particles[idx].energy = scalars[idx].energy;
            particles[idx].kinetic_energy = scalars[idx].kinetic_energy;
            particles[idx].mass = scalars[idx].mass;
            particles[idx].entropy_bits = scalars[idx].entropy_bits;
        }

        drop(pos_data);
        drop(vel_data);
        drop(scalars_data);
        self.positions_staging.unmap();
        self.velocities_staging.unmap();
        self.scalars_staging.unmap();

        particles
    }

    /// Batch read particles and accept flags in a single GPU round-trip
    pub fn read_particles_and_flags(&self) -> (Vec<ThermodynamicParticle>, Vec<u32>) {
        let pos_vel_size = (self.dim * self.particle_count * std::mem::size_of::<f16>()) as u64;
        let scalars_size = (self.particle_count * std::mem::size_of::<ParticleScalars>()) as u64;
        let flags_size = (self.particle_count * 4) as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batch_read_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.positions_buffer,
            0,
            &self.positions_staging,
            0,
            pos_vel_size,
        );
        encoder.copy_buffer_to_buffer(
            &self.velocities_buffer,
            0,
            &self.velocities_staging,
            0,
            pos_vel_size,
        );
        encoder.copy_buffer_to_buffer(
            &self.scalars_buffer,
            0,
            &self.scalars_staging,
            0,
            scalars_size,
        );
        encoder.copy_buffer_to_buffer(
            &self.accept_flags_buffer,
            0,
            &self.accept_flags_staging,
            0,
            flags_size,
        );
        self.queue.submit(Some(encoder.finish()));

        // Map all buffers
        let pos_slice = self.positions_staging.slice(..pos_vel_size);
        let vel_slice = self.velocities_staging.slice(..pos_vel_size);
        let scalars_slice = self.scalars_staging.slice(..scalars_size);
        let flags_slice = self.accept_flags_staging.slice(..flags_size);

        let (tx1, rx1) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        let (tx2, rx2) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        let (tx3, rx3) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        let (tx4, rx4) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();

        pos_slice.map_async(wgpu::MapMode::Read, move |r| {
            tx1.send(r).unwrap();
        });
        vel_slice.map_async(wgpu::MapMode::Read, move |r| {
            tx2.send(r).unwrap();
        });
        scalars_slice.map_async(wgpu::MapMode::Read, move |r| {
            tx3.send(r).unwrap();
        });
        flags_slice.map_async(wgpu::MapMode::Read, move |r| {
            tx4.send(r).unwrap();
        });

        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        rx1.recv().unwrap().expect("positions map failed");
        rx2.recv().unwrap().expect("velocities map failed");
        rx3.recv().unwrap().expect("scalars map failed");
        rx4.recv().unwrap().expect("flags map failed");

        // Read data
        let pos_data = pos_slice.get_mapped_range();
        let vel_data = vel_slice.get_mapped_range();
        let scalars_data = scalars_slice.get_mapped_range();
        let flags_data = flags_slice.get_mapped_range();

        let positions: &[f16] = bytemuck::cast_slice(&pos_data);
        let velocities: &[f16] = bytemuck::cast_slice(&vel_data);
        let scalars: &[ParticleScalars] = bytemuck::cast_slice(&scalars_data);
        let flags: Vec<u32> = bytemuck::cast_slice(&flags_data).to_vec();

        // Reconstruct particles from SoA (dimension-major) to AoS
        let mut particles = vec![ThermodynamicParticle::default(); self.particle_count];
        for idx in 0..self.particle_count {
            for d in 0..self.dim {
                particles[idx].pos[d] = positions[d * self.particle_count + idx];
                particles[idx].vel[d] = velocities[d * self.particle_count + idx];
            }
            particles[idx].energy = scalars[idx].energy;
            particles[idx].kinetic_energy = scalars[idx].kinetic_energy;
            particles[idx].mass = scalars[idx].mass;
            particles[idx].entropy_bits = scalars[idx].entropy_bits;
        }

        drop(pos_data);
        drop(vel_data);
        drop(scalars_data);
        drop(flags_data);
        self.positions_staging.unmap();
        self.velocities_staging.unmap();
        self.scalars_staging.unmap();
        self.accept_flags_staging.unmap();

        (particles, flags)
    }

    /// Extract entropy (only valid in entropy mode)
    pub fn extract_entropy(&mut self) -> Vec<u32> {
        if self.mode() != ThermodynamicMode::Entropy {
            return Vec::new();
        }

        let size = (self.particle_count * 4) as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("entropy_encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.entropy_buffer, 0, &self.entropy_staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.entropy_staging.slice(..size);
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

        let data = slice.get_mapped_range();
        let entropy: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.entropy_staging.unmap();

        entropy
    }

    /// Compute statistics about current particle distribution
    pub fn statistics(&self) -> ThermodynamicStats {
        let particles = self.read_particles();

        let energies: Vec<f32> = particles.iter().map(|p| p.energy).collect();
        let mean_energy = energies.iter().sum::<f32>() / particles.len() as f32;
        let min_energy = energies.iter().cloned().fold(f32::MAX, f32::min);
        let max_energy = energies.iter().cloned().fold(f32::MIN, f32::max);

        let mut spread = 0.0;
        let mut count = 0;
        for i in 0..particles.len().min(100) {
            for j in (i + 1)..particles.len().min(100) {
                let dx = particles[i].pos[0].to_f32() - particles[j].pos[0].to_f32();
                let dy = particles[i].pos[1].to_f32() - particles[j].pos[1].to_f32();
                spread += (dx * dx + dy * dy).sqrt();
                count += 1;
            }
        }
        spread /= count.max(1) as f32;

        let low_energy_count = energies.iter().filter(|&&e| e < 0.1).count();

        ThermodynamicStats {
            mean_energy,
            min_energy,
            max_energy,
            spread,
            low_energy_fraction: low_energy_count as f32 / particles.len() as f32,
            mode: self.mode(),
            temperature: self.temperature,
        }
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    pub fn particle_count(&self) -> usize {
        self.particle_count
    }

    /// Compute comprehensive diversity metrics for the current population
    pub fn diversity_metrics(&self) -> DiversityMetrics {
        let particles = self.read_particles();
        let n = particles.len();
        let dim = self.dim;

        let sample_size = n.min(500);
        let step = if n > sample_size { n / sample_size } else { 1 };
        let sampled: Vec<_> = particles.iter().step_by(step).take(sample_size).collect();
        let m = sampled.len();

        let mut distances = Vec::new();
        for i in 0..m {
            for j in (i + 1)..m {
                let mut dist_sq = 0.0f32;
                for d in 0..dim {
                    let diff = sampled[i].pos[d].to_f32() - sampled[j].pos[d].to_f32();
                    dist_sq += diff * diff;
                }
                distances.push(dist_sq.sqrt());
            }
        }

        let mean_dist = if !distances.is_empty() {
            distances.iter().sum::<f32>() / distances.len() as f32
        } else {
            0.0
        };

        let dist_var = if !distances.is_empty() {
            distances
                .iter()
                .map(|d| (d - mean_dist).powi(2))
                .sum::<f32>()
                / distances.len() as f32
        } else {
            0.0
        };

        let energies: Vec<f32> = particles
            .iter()
            .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
            .map(|p| p.energy)
            .collect();

        let mean_energy = energies.iter().sum::<f32>() / energies.len().max(1) as f32;
        let energy_var = energies
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f32>()
            / energies.len().max(1) as f32;

        let min_energy = energies.iter().cloned().fold(f32::MAX, f32::min);
        let weights: Vec<f32> = energies
            .iter()
            .map(|e| (-(e - min_energy) / self.temperature.max(0.001)).exp())
            .collect();
        let sum_w: f32 = weights.iter().sum();
        let sum_w_sq: f32 = weights.iter().map(|w| w * w).sum();
        let ess = if sum_w_sq > 0.0 {
            (sum_w * sum_w / sum_w_sq).min(n as f32)
        } else {
            1.0
        };

        let energy_threshold = min_energy + energy_var.sqrt();
        let low_energy: Vec<_> = sampled
            .iter()
            .filter(|p| !p.energy.is_nan() && p.energy < energy_threshold)
            .collect();

        let mut modes = 0usize;
        let mode_dist_threshold = mean_dist * 0.3;
        let mut mode_centers: Vec<Vec<f32>> = Vec::new();

        for p in &low_energy {
            let pos: Vec<f32> = (0..dim).map(|d| p.pos[d].to_f32()).collect();
            let is_new_mode = mode_centers.iter().all(|center| {
                let dist: f32 = pos
                    .iter()
                    .zip(center.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                dist > mode_dist_threshold
            });
            if is_new_mode {
                modes += 1;
                mode_centers.push(pos);
            }
        }

        let mut min_pos = vec![f32::MAX; dim];
        let mut max_pos = vec![f32::MIN; dim];
        for p in &sampled {
            for d in 0..dim {
                let v = p.pos[d].to_f32();
                min_pos[d] = min_pos[d].min(v);
                max_pos[d] = max_pos[d].max(v);
            }
        }

        let expected_range = 8.0f32;
        let mut coverage = 1.0f32;
        for d in 0..dim.min(8) {
            let range = (max_pos[d] - min_pos[d]).max(0.001);
            coverage *= (range / expected_range).min(1.0);
        }
        coverage = coverage.powf(1.0 / dim.min(8) as f32);

        DiversityMetrics {
            mean_pairwise_distance: mean_dist,
            distance_std: dist_var.sqrt(),
            energy_variance: energy_var,
            effective_sample_size: ess,
            estimated_modes: modes.max(1),
            coverage,
            dim,
        }
    }

    // ========================================================================
    // HMC PARAMETER ACCESSORS
    // ========================================================================

    /// Set the number of leapfrog steps per HMC iteration
    pub fn set_leapfrog_steps(&mut self, steps: u32) {
        self.leapfrog_steps = steps;
    }

    /// Get current leapfrog steps
    pub fn leapfrog_steps(&self) -> u32 {
        self.leapfrog_steps
    }

    /// Set the leapfrog step size (ε)
    pub fn set_step_size(&mut self, step_size: f32) {
        self.step_size = step_size;
    }

    /// Get current step size
    pub fn step_size(&self) -> f32 {
        self.step_size
    }

    /// Set particle mass
    pub fn set_mass(&mut self, mass: f32) {
        self.mass = mass;
    }

    /// Get current mass
    pub fn mass(&self) -> f32 {
        self.mass
    }

    /// Read acceptance flags from the last step
    pub fn read_accept_flags(&self) -> Vec<u32> {
        let size = (self.particle_count * 4) as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("accept_flags_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.accept_flags_buffer,
            0,
            &self.accept_flags_staging,
            0,
            size,
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.accept_flags_staging.slice(..size);
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

        let data = slice.get_mapped_range();
        let flags: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.accept_flags_staging.unmap();

        flags
    }

    /// Get the acceptance rate from the last step
    /// Returns a value between 0.0 and 1.0
    pub fn acceptance_rate(&self) -> f32 {
        let flags = self.read_accept_flags();
        let accepted: u32 = flags.iter().sum();
        accepted as f32 / flags.len() as f32
    }

    /// Adapt step size to target a specific acceptance rate (typically ~0.65 for HMC)
    /// Call this after step() to tune the step size
    pub fn adapt_step_size(&mut self, target_rate: f32) {
        let current_rate = self.acceptance_rate();
        if current_rate < target_rate - 0.1 {
            // Acceptance too low, reduce step size
            self.step_size *= 0.9;
        } else if current_rate > target_rate + 0.1 {
            // Acceptance too high, increase step size
            self.step_size *= 1.1;
        }
        // Clamp to reasonable range
        self.step_size = self.step_size.clamp(0.001, 1.0);
    }
}
