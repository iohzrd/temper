//! Unified Thermodynamic Particle System
//!
//! Demonstrates that entropy generation, Bayesian sampling, and optimization
//! are all the same algorithm with different temperature settings:
//!
//! - T >> 1.0  : Entropy mode - chaotic exploration, extract random bits
//! - T ~ 0.1   : Sampling mode - SVGD/Langevin samples from posterior
//! - T → 0    : Optimize mode - gradient descent to minima
//!
//! This is the core thesis: thermodynamic computation as a unifying framework.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Operating mode determined by temperature
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermodynamicMode {
    Optimize,  // T < 0.01: Pure gradient descent
    Sample,    // 0.01 <= T <= 1.0: Bayesian sampling
    Entropy,   // T > 1.0: Entropy extraction
}

/// Loss function to optimize
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(u32)]
pub enum LossFunction {
    #[default]
    NeuralNet2D = 0,  // Original 2D neural net
    Multimodal = 1,    // N-dimensional multimodal
    Rosenbrock = 2,    // Classic banana valley, min at (1,1,...,1)
    Rastrigin = 3,     // Highly multimodal, min at origin
    Ackley = 4,        // Flat outer region, hole at center
    Sphere = 5,        // Simple convex baseline
    MlpXor = 6,        // Real MLP on XOR problem (9 params)
    MlpSpiral = 7,     // Real MLP on spiral classification
    MlpDeep = 8,       // Deep MLP: 2->4->4->1 (37 params) on circles dataset
    Schwefel = 9,      // Deceptive - global min at (420.97, ...) far from origin
    Custom = 10,       // Custom expression-based loss function
}

impl ThermodynamicMode {
    pub fn from_temperature(t: f32) -> Self {
        if t < 0.01 {
            Self::Optimize
        } else if t <= 1.0 {
            Self::Sample
        } else {
            Self::Entropy
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Optimize => "OPTIMIZE",
            Self::Sample => "SAMPLE",
            Self::Entropy => "ENTROPY",
        }
    }
}

const MAX_DIM: usize = 64;
const MAX_PARTICLES: usize = 500_000;  // Support up to 500k particles (GPU memory dependent)

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ThermodynamicParticle {
    pub pos: [f32; MAX_DIM],
    pub vel: [f32; MAX_DIM],
    pub energy: f32,
    pub entropy_bits: u32,
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    particle_count: u32,
    dim: u32,
    gamma: f32,
    temperature: f32,
    repulsion_strength: f32,
    kernel_bandwidth: f32,
    dt: f32,
    seed: u32,
    mode: u32,
    loss_fn: u32,
    repulsion_samples: u32,  // 0 = skip repulsion, >0 = sample K particles (O(nK) instead of O(n²))
    _pad: f32,
}

/// Unified thermodynamic particle system
pub struct ThermodynamicSystem {
    device: wgpu::Device,
    queue: wgpu::Queue,
    particle_buffer: wgpu::Buffer,
    repulsion_buffer: wgpu::Buffer,
    entropy_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    entropy_staging: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    repulsion_pipeline: wgpu::ComputePipeline,
    update_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    particle_count: usize,
    dim: usize,
    temperature: f32,
    gamma: f32,
    repulsion_strength: f32,
    kernel_bandwidth: f32,
    dt: f32,
    step: u32,
    loss_fn: LossFunction,
    repulsion_samples: u32,  // 0 = skip, 64 = sample 64 particles (default)
    // Entropy extraction
    entropy_pool: Vec<u32>,
    // Custom loss function WGSL code (if using LossFunction::Custom)
    #[allow(dead_code)]
    custom_loss_wgsl: Option<String>,
}

impl ThermodynamicSystem {
    pub fn new(
        particle_count: usize,
        dim: usize,
        temperature: f32,
    ) -> Self {
        assert!(particle_count <= MAX_PARTICLES);
        assert!(dim <= MAX_DIM);

        // Initialize particles randomly in [-4, 4]
        let mut particles = vec![ThermodynamicParticle {
            pos: [0.0; MAX_DIM],
            vel: [0.0; MAX_DIM],
            energy: 0.0,
            entropy_bits: 0,
            _pad: [0.0; 2],
        }; particle_count];

        let mut seed = 42u64;
        for p in particles.iter_mut() {
            for d in 0..dim {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                p.pos[d] = -4.0 + (seed & 0xFFFF) as f32 / 65535.0 * 8.0;
            }
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

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor::default(),
        ))
        .expect("Failed to create device");

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particles"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let repulsion_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repulsion"),
            size: (particle_count * std::mem::size_of::<ThermodynamicParticle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let entropy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entropy"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (particle_count * std::mem::size_of::<ThermodynamicParticle>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let entropy_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entropy_staging"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Adaptive parameters based on temperature
        let (gamma, repulsion_strength, kernel_bandwidth, dt) = Self::params_for_temperature(temperature, dim);

        // Default repulsion samples: 64 for sampling/entropy mode, 0 for optimize mode
        let repulsion_samples = if temperature < 0.01 { 0 } else { 64 };

        let uniforms = Uniforms {
            particle_count: particle_count as u32,
            dim: dim as u32,
            gamma,
            temperature,
            repulsion_strength,
            kernel_bandwidth,
            dt,
            seed: 12345,
            mode: ThermodynamicMode::from_temperature(temperature) as u32,
            loss_fn: LossFunction::default() as u32,
            repulsion_samples,
            _pad: 0.0,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("thermodynamic"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/thermodynamic.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("thermodynamic_layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thermodynamic_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: repulsion_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: entropy_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("thermodynamic_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let repulsion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("repulsion_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_repulsion"),
            compilation_options: Default::default(),
            cache: None,
        });

        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("update_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            particle_buffer,
            repulsion_buffer,
            entropy_buffer,
            staging_buffer,
            entropy_staging,
            uniform_buffer,
            repulsion_pipeline,
            update_pipeline,
            bind_group,
            particle_count,
            dim,
            temperature,
            gamma,
            repulsion_strength,
            kernel_bandwidth,
            dt,
            step: 0,
            loss_fn: LossFunction::default(),
            repulsion_samples,
            entropy_pool: Vec::new(),
            custom_loss_wgsl: None,
        }
    }

    /// Create with a specific loss function
    pub fn with_loss_function(
        particle_count: usize,
        dim: usize,
        temperature: f32,
        loss_fn: LossFunction,
    ) -> Self {
        let mut sys = Self::new(particle_count, dim, temperature);
        sys.set_loss_function(loss_fn);
        sys
    }

    /// Create with a custom expression-based loss function
    ///
    /// # Example
    /// ```ignore
    /// use temper::expr::*;
    /// use temper::ThermodynamicSystem;
    ///
    /// // Griewank function
    /// let griewank = const_(1.0)
    ///     + sum_dims(|x, _| x.powi(2) / 4000.0)
    ///     - prod_dims(|x, i| cos(x / sqrt(i + 1.0)));
    ///
    /// let mut system = ThermodynamicSystem::with_expr(
    ///     500, 4, 1.0, griewank
    /// );
    /// ```
    pub fn with_expr(
        particle_count: usize,
        dim: usize,
        temperature: f32,
        expr: crate::expr::Expr,
    ) -> Self {
        Self::with_expr_options(particle_count, dim, temperature, expr, true)
    }

    /// Create with a custom expression and control gradient type
    ///
    /// # Arguments
    /// * `analytical_gradients` - If true, use symbolic differentiation (faster).
    ///   If false, use numerical finite differences (slower but always works).
    pub fn with_expr_options(
        particle_count: usize,
        dim: usize,
        temperature: f32,
        expr: crate::expr::Expr,
        analytical_gradients: bool,
    ) -> Self {
        assert!(particle_count <= MAX_PARTICLES);
        assert!(dim <= MAX_DIM);

        // Initialize particles randomly in [-4, 4]
        let mut particles = vec![ThermodynamicParticle {
            pos: [0.0; MAX_DIM],
            vel: [0.0; MAX_DIM],
            energy: 0.0,
            entropy_bits: 0,
            _pad: [0.0; 2],
        }; particle_count];

        let mut seed = 42u64;
        for p in particles.iter_mut() {
            for d in 0..dim {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                p.pos[d] = -4.0 + (seed & 0xFFFF) as f32 / 65535.0 * 8.0;
            }
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

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor::default(),
        ))
        .expect("Failed to create device");

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particles"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let repulsion_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repulsion"),
            size: (particle_count * std::mem::size_of::<ThermodynamicParticle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let entropy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entropy"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (particle_count * std::mem::size_of::<ThermodynamicParticle>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let entropy_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entropy_staging"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (gamma, repulsion_strength, kernel_bandwidth, dt) = Self::params_for_temperature(temperature, dim);
        let repulsion_samples = if temperature < 0.01 { 0 } else { 64 };

        let uniforms = Uniforms {
            particle_count: particle_count as u32,
            dim: dim as u32,
            gamma,
            temperature,
            repulsion_strength,
            kernel_bandwidth,
            dt,
            seed: 12345,
            mode: ThermodynamicMode::from_temperature(temperature) as u32,
            loss_fn: LossFunction::Custom as u32,
            repulsion_samples,
            _pad: 0.0,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Generate custom shader by replacing stub functions with real ones
        let custom_wgsl = expr.to_wgsl_with_options(analytical_gradients);
        let base_shader = include_str!("shaders/thermodynamic.wgsl");

        // Find and replace the stub functions
        let stub_marker = "// Stub functions for custom expressions";
        let stub_end = "// 2D neural net:";

        let full_shader = if let Some(stub_start) = base_shader.find(stub_marker) {
            if let Some(stub_end_pos) = base_shader[stub_start..].find(stub_end) {
                // Replace stubs with custom implementation
                let before = &base_shader[..stub_start];
                let after = &base_shader[stub_start + stub_end_pos..];
                format!("{}{}\n\n{}", before, custom_wgsl, after)
            } else {
                // Fallback: prepend
                format!("{}\n\n{}", custom_wgsl, base_shader)
            }
        } else {
            // Fallback: prepend
            format!("{}\n\n{}", custom_wgsl, base_shader)
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("thermodynamic_custom"),
            source: wgpu::ShaderSource::Wgsl(full_shader.clone().into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("thermodynamic_layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thermodynamic_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: repulsion_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: entropy_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("thermodynamic_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let repulsion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("repulsion_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_repulsion"),
            compilation_options: Default::default(),
            cache: None,
        });

        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("update_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            particle_buffer,
            repulsion_buffer,
            entropy_buffer,
            staging_buffer,
            entropy_staging,
            uniform_buffer,
            repulsion_pipeline,
            update_pipeline,
            bind_group,
            particle_count,
            dim,
            temperature,
            gamma,
            repulsion_strength,
            kernel_bandwidth,
            dt,
            step: 0,
            loss_fn: LossFunction::Custom,
            repulsion_samples,
            entropy_pool: Vec::new(),
            custom_loss_wgsl: Some(full_shader),
        }
    }

    /// Get adaptive parameters for a given temperature
    fn params_for_temperature(temperature: f32, dim: usize) -> (f32, f32, f32, f32) {
        let mode = ThermodynamicMode::from_temperature(temperature);
        match mode {
            ThermodynamicMode::Optimize => {
                // Pure optimization: high gamma (strong gradient following), no repulsion
                (1.0, 0.0, 0.5, 0.1)
            }
            ThermodynamicMode::Sample => {
                // Sampling: balanced parameters for SVGD
                let bandwidth = 0.3 + 0.1 * dim as f32; // Scale with dimension
                (0.5, 0.2, bandwidth, 0.01)
            }
            ThermodynamicMode::Entropy => {
                // Entropy: low gamma (less gradient influence), high noise
                (0.1, 0.05, 1.0, 0.05)
            }
        }
    }

    /// Set temperature and adapt parameters
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
        let (gamma, repulsion_strength, kernel_bandwidth, dt) =
            Self::params_for_temperature(temperature, self.dim);
        self.gamma = gamma;
        self.repulsion_strength = repulsion_strength;
        self.kernel_bandwidth = kernel_bandwidth;
        self.dt = dt;
        // Automatically adjust repulsion samples based on mode
        // 0 for optimization (no repulsion needed), 64 for sampling/entropy
        self.repulsion_samples = if temperature < 0.01 { 0 } else { 64 };
    }

    /// Set the number of repulsion samples (0 = skip repulsion, >0 = sample K particles)
    /// This controls the O(nK) vs O(n²) tradeoff:
    /// - 0: Skip repulsion entirely (fastest, use for pure optimization)
    /// - 64: Good default for most sampling tasks
    /// - particle_count: Full O(n²) computation (most accurate but slowest)
    pub fn set_repulsion_samples(&mut self, samples: u32) {
        self.repulsion_samples = samples;
    }

    /// Get current repulsion samples setting
    pub fn repulsion_samples(&self) -> u32 {
        self.repulsion_samples
    }

    /// Set the time step (dt) for finer control over stability
    /// Smaller dt = more stable but slower convergence
    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt;
    }

    /// Get current time step
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// Set the loss function
    pub fn set_loss_function(&mut self, loss_fn: LossFunction) {
        self.loss_fn = loss_fn;
    }

    /// Get current loss function
    pub fn loss_function(&self) -> LossFunction {
        self.loss_fn
    }

    /// Get current operating mode
    pub fn mode(&self) -> ThermodynamicMode {
        ThermodynamicMode::from_temperature(self.temperature)
    }

    /// Run one simulation step
    pub fn step(&mut self) {
        self.step += 1;

        let uniforms = Uniforms {
            particle_count: self.particle_count as u32,
            dim: self.dim as u32,
            gamma: self.gamma,
            temperature: self.temperature,
            repulsion_strength: self.repulsion_strength,
            kernel_bandwidth: self.kernel_bandwidth,
            dt: self.dt,
            seed: self.step * 1337,
            mode: self.mode() as u32,
            loss_fn: self.loss_fn as u32,
            repulsion_samples: self.repulsion_samples,
            _pad: 0.0,
        };
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("thermodynamic_encoder"),
        });

        let workgroups = (self.particle_count as u32 + 63) / 64;

        // Pass 1: Compute repulsion
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("repulsion_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.repulsion_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 2: Update particles
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("update_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Read particles back from GPU
    pub fn read_particles(&self) -> Vec<ThermodynamicParticle> {
        let size = (self.particle_count * std::mem::size_of::<ThermodynamicParticle>()) as u64;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.particle_buffer, 0, &self.staging_buffer, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buffer.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        rx.recv().unwrap().expect("map failed");

        let data = slice.get_mapped_range();
        let particles: Vec<ThermodynamicParticle> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_buffer.unmap();

        particles
    }

    /// Extract entropy (only valid in entropy mode)
    pub fn extract_entropy(&mut self) -> Vec<u32> {
        if self.mode() != ThermodynamicMode::Entropy {
            return Vec::new();
        }

        let size = (self.particle_count * 4) as u64;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("entropy_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.entropy_buffer, 0, &self.entropy_staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.entropy_staging.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
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

        // Compute spread (average pairwise distance in first 2 dims)
        let mut spread = 0.0;
        let mut count = 0;
        for i in 0..particles.len().min(100) {
            for j in (i + 1)..particles.len().min(100) {
                let dx = particles[i].pos[0] - particles[j].pos[0];
                let dy = particles[i].pos[1] - particles[j].pos[1];
                spread += (dx * dx + dy * dy).sqrt();
                count += 1;
            }
        }
        spread /= count.max(1) as f32;

        // Count particles near minima (low energy)
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
}

#[derive(Debug, Clone)]
pub struct ThermodynamicStats {
    pub mean_energy: f32,
    pub min_energy: f32,
    pub max_energy: f32,
    pub spread: f32,
    pub low_energy_fraction: f32,
    pub mode: ThermodynamicMode,
    pub temperature: f32,
}

/// Adaptive temperature scheduler for simulated annealing
///
/// Adjusts cooling rate based on optimization progress:
/// - Convergence detection: cool faster when near optimum
/// - Stall detection: slow down or reheat when stuck
/// - Dimension-aware: parameters scale with problem dimension
///
/// # Example
/// ```ignore
/// use temper::{ThermodynamicSystem, AdaptiveScheduler, LossFunction};
///
/// let mut system = ThermodynamicSystem::with_loss_function(
///     500, 8, 5.0, LossFunction::Rastrigin
/// );
/// let mut scheduler = AdaptiveScheduler::new(5.0, 0.001, 0.8, 8);
///
/// for _ in 0..5000 {
///     let particles = system.read_particles();
///     let min_energy = particles.iter()
///         .filter(|p| !p.energy.is_nan())
///         .map(|p| p.energy)
///         .fold(f32::MAX, f32::min);
///
///     let temp = scheduler.update(min_energy);
///     system.set_temperature(temp);
///     system.step();
/// }
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveScheduler {
    temperature: f32,
    t_start: f32,
    t_end: f32,
    base_cooling_rate: f32,
    energy_history: Vec<f32>,
    stall_count: u32,
    reheat_count: u32,
    convergence_threshold: f32,
    dim: usize,
    max_reheats: u32,
    total_steps: u32,
}

impl AdaptiveScheduler {
    /// Create a new adaptive scheduler
    ///
    /// # Arguments
    /// * `t_start` - Starting temperature (high = more exploration)
    /// * `t_end` - Ending temperature (low = pure optimization)
    /// * `convergence_threshold` - Energy below which we consider converged
    /// * `dim` - Problem dimensionality (affects parameter scaling)
    pub fn new(t_start: f32, t_end: f32, convergence_threshold: f32, dim: usize) -> Self {
        Self::with_steps(t_start, t_end, convergence_threshold, dim, 5000)
    }

    /// Create with custom step count for cooling rate calculation
    pub fn with_steps(t_start: f32, t_end: f32, convergence_threshold: f32, dim: usize, total_steps: u32) -> Self {
        let base_cooling_rate = (t_end / t_start).powf(1.0 / total_steps as f32);
        // Limit reheats based on dimension - high-D needs stability more than exploration
        let max_reheats = if dim <= 4 { 20 } else if dim <= 8 { 10 } else { 5 };

        Self {
            temperature: t_start,
            t_start,
            t_end,
            base_cooling_rate,
            energy_history: Vec::new(),
            stall_count: 0,
            reheat_count: 0,
            convergence_threshold,
            dim,
            max_reheats,
            total_steps,
        }
    }

    /// Update temperature based on current minimum energy
    ///
    /// Call this once per step with the best energy found so far.
    /// Returns the new temperature to use.
    pub fn update(&mut self, min_energy: f32) -> f32 {
        self.energy_history.push(min_energy);

        // Convergence detection - cool fast when near optimum
        if min_energy < self.convergence_threshold {
            self.temperature *= self.base_cooling_rate.powf(2.0);
            self.stall_count = 0;
            self.temperature = self.temperature.max(self.t_end);
            return self.temperature;
        }

        // Longer window for high-D (slow progress is normal)
        let window = 50 + self.dim * 5;
        if self.energy_history.len() < window {
            self.temperature *= self.base_cooling_rate;
            return self.temperature;
        }

        let recent_start = self.energy_history.len() - window;
        let old_energy = self.energy_history[recent_start];
        let improvement = old_energy - min_energy;
        let improvement_rate = if old_energy > 0.01 { improvement / old_energy } else { improvement };

        // Scale stall threshold with dimension - high-D progress is naturally slower
        let dim_factor = (self.dim as f32).sqrt() / 1.4;
        let stall_threshold = if min_energy > 100.0 { 0.0005 / dim_factor }
                             else if min_energy > 10.0 { 0.002 / dim_factor }
                             else if min_energy > 1.0 { 0.005 / dim_factor }
                             else { 0.02 / dim_factor };

        // Require longer stall before reheating in high-D
        let stall_required = 80 + self.dim as u32 * 10;

        if improvement_rate < stall_threshold && self.temperature > self.t_end * 10.0 {
            self.stall_count += 1;
            if self.stall_count > stall_required
               && min_energy > self.convergence_threshold * 10.0
               && self.reheat_count < self.max_reheats {
                // Smaller reheat factor in high-D
                let reheat_factor = if self.dim <= 4 { 2.0 } else { 1.3 };
                self.temperature = (self.temperature * reheat_factor).min(self.t_start * 0.3);
                self.stall_count = 0;
                self.reheat_count += 1;
            } else {
                self.temperature *= self.base_cooling_rate.powf(0.5);
            }
        } else if improvement_rate > stall_threshold * 3.0 {
            self.temperature *= self.base_cooling_rate.powf(1.5);
            self.stall_count = 0;
        } else {
            self.temperature *= self.base_cooling_rate;
            self.stall_count = self.stall_count.saturating_sub(1);
        }

        self.temperature = self.temperature.max(self.t_end);
        self.temperature
    }

    /// Reset the scheduler to initial state
    pub fn reset(&mut self) {
        self.temperature = self.t_start;
        self.energy_history.clear();
        self.stall_count = 0;
        self.reheat_count = 0;
    }

    /// Get current temperature
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Get number of reheats performed
    pub fn reheat_count(&self) -> u32 {
        self.reheat_count
    }

    /// Get the energy history
    pub fn energy_history(&self) -> &[f32] {
        &self.energy_history
    }

    /// Check if scheduler detected convergence
    pub fn is_converged(&self) -> bool {
        self.energy_history.last().map_or(false, |&e| e < self.convergence_threshold)
    }

    /// Get progress as fraction (0.0 to 1.0) based on temperature
    pub fn progress(&self) -> f32 {
        let log_t = (self.temperature.ln() - self.t_end.ln()) / (self.t_start.ln() - self.t_end.ln());
        1.0 - log_t.clamp(0.0, 1.0)
    }
}
