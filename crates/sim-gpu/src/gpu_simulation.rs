use common::{Dim2, Dimension, SimConfig};
use sim_core::obstacle::Obstacle;
use sim_core::particle::ParticleStorage;
use sim_core::scene::{self, SceneDescription};

use sim_core::neighbor::SpatialHashGrid;

use crate::buffers::ParticleBuffers;
use crate::context::GpuContext;
use crate::grid_buffers::GridBuffers;
use crate::pipelines::clear_gravity::ClearGravityPipeline;
use crate::pipelines::density::DensityPipeline;
use crate::pipelines::forces::ForcesPipeline;
use crate::pipelines::integrate::IntegratePipeline;
use crate::pipelines::pressure::PressurePipeline;
use crate::uniforms::{ObstacleArrayUniform, SimParamsUniform};

/// GPU-accelerated simulation backend.
/// Mirrors the CPU `Simulation<Dim2>` but runs compute shaders on the GPU.
pub struct GpuSimulation {
    context: GpuContext,
    buffers: ParticleBuffers,
    params_buffer: wgpu::Buffer,
    obstacles_buffer: wgpu::Buffer,
    grid_buffers: GridBuffers,
    grid: SpatialHashGrid<Dim2>,
    clear_gravity_pipeline: ClearGravityPipeline,
    density_pipeline: DensityPipeline,
    pressure_pipeline: PressurePipeline,
    forces_pipeline: ForcesPipeline,
    integrate_pipeline: IntegratePipeline,
    pub config: SimConfig,
    pub time: f32,
    pub step_count: u64,
    pub obstacles: Vec<Obstacle<Dim2>>,
    /// CPU-side particle storage for spawn/erase/reset operations.
    cpu_particles: ParticleStorage<Dim2>,
    /// Whether CPU particles have been modified and need upload.
    dirty: bool,
    /// Current particle count on GPU.
    particle_count: u32,
}

impl GpuSimulation {
    /// Create a new GPU simulation from an existing GPU context.
    pub fn new(context: GpuContext, config: SimConfig) -> Self {
        let device = &context.device;

        let buffers = ParticleBuffers::new(device, 1024);

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim_params"),
            size: std::mem::size_of::<SimParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let obstacles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("obstacles"),
            size: std::mem::size_of::<ObstacleArrayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_buffers = GridBuffers::new(device, 1024, common::GRID_TABLE_SIZE);
        let grid = SpatialHashGrid::new(2.0 * config.smoothing_radius);
        let clear_gravity_pipeline = ClearGravityPipeline::new(device);
        let density_pipeline = DensityPipeline::new(device);
        let pressure_pipeline = PressurePipeline::new(device);
        let forces_pipeline = ForcesPipeline::new(device);
        let integrate_pipeline = IntegratePipeline::new(device);

        Self {
            context,
            buffers,
            params_buffer,
            obstacles_buffer,
            grid_buffers,
            grid,
            clear_gravity_pipeline,
            density_pipeline,
            pressure_pipeline,
            forces_pipeline,
            integrate_pipeline,
            config,
            time: 0.0,
            step_count: 0,
            obstacles: Vec::new(),
            cpu_particles: ParticleStorage::new(),
            dirty: false,
            particle_count: 0,
        }
    }

    /// Create a GpuSimulation by initializing a new GPU context.
    pub fn create(config: SimConfig) -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let context = pollster::block_on(GpuContext::new(&instance, None))?;
        Some(Self::new(context, config))
    }

    pub fn particle_count(&self) -> usize {
        self.particle_count as usize
    }

    /// Advance the simulation by one time step.
    pub fn step(&mut self) {
        if self.particle_count == 0 && self.cpu_particles.is_empty() {
            self.time += self.config.time_step;
            self.step_count += 1;
            return;
        }

        let device = &self.context.device;
        let queue = &self.context.queue;

        // Upload CPU particles if dirty
        if self.dirty {
            self.buffers
                .upload_from_storage(device, queue, &self.cpu_particles);
            self.particle_count = self.cpu_particles.len() as u32;
            self.dirty = false;
        }

        if self.particle_count == 0 {
            self.time += self.config.time_step;
            self.step_count += 1;
            return;
        }

        // Update uniform buffers
        let params = SimParamsUniform::from_config(&self.config, self.particle_count);
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let obstacles_uniform = ObstacleArrayUniform::from_obstacles(&self.obstacles);
        queue.write_buffer(
            &self.obstacles_buffer,
            0,
            bytemuck::bytes_of(&obstacles_uniform),
        );

        // CPU grid build: download positions, build grid, upload to GPU.
        // Known performance limitation: GPU→CPU→GPU round-trip each step.
        // Future: GPU-side grid build (planned for M6).
        let positions_cpu = self.buffers.download_positions(device, queue, self.particle_count);
        {
            let positions_nalgebra: Vec<nalgebra::Vector2<f32>> = positions_cpu
                .iter()
                .map(|p| nalgebra::Vector2::new(p[0], p[1]))
                .collect();
            self.grid
                .set_cell_size(2.0 * self.config.smoothing_radius);
            self.grid.build(&positions_nalgebra);
            let (sorted_indices, cell_starts, cell_counts) = self.grid.export_for_gpu();
            self.grid_buffers
                .upload(device, queue, &sorted_indices, &cell_starts, &cell_counts);
        }

        // Create bind groups
        let clear_gravity_bg = self.clear_gravity_pipeline.create_bind_group(
            device,
            &self.params_buffer,
            &self.buffers.accelerations,
        );

        let density_particle_bg = self.density_pipeline.create_particle_bind_group(
            device,
            &self.params_buffer,
            &self.buffers.positions,
            &self.buffers.masses,
            &self.buffers.densities,
        );
        let density_grid_bg = self.density_pipeline.create_grid_bind_group(
            device,
            &self.grid_buffers.sorted_indices,
            &self.grid_buffers.cell_start,
            &self.grid_buffers.cell_count,
        );

        let pressure_bg = self.pressure_pipeline.create_bind_group(
            device,
            &self.params_buffer,
            &self.buffers.densities,
            &self.buffers.pressures,
        );

        let forces_particle_bg = self.forces_pipeline.create_particle_bind_group(
            device,
            &self.params_buffer,
            &self.buffers.positions,
            &self.buffers.velocities,
            &self.buffers.densities,
            &self.buffers.pressures,
            &self.buffers.masses,
            &self.buffers.accelerations,
        );
        let forces_grid_bg = self.forces_pipeline.create_grid_bind_group(
            device,
            &self.grid_buffers.sorted_indices,
            &self.grid_buffers.cell_start,
            &self.grid_buffers.cell_count,
        );

        let integrate_bg = self.integrate_pipeline.create_bind_group(
            device,
            &self.params_buffer,
            &self.buffers.positions,
            &self.buffers.velocities,
            &self.buffers.accelerations,
            &self.obstacles_buffer,
        );

        // Encode full SPH pipeline. Each stage uses a separate compute pass
        // so storage buffer writes are synchronized between dependent stages.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim_step_encoder"),
        });

        // 1. Clear accelerations + apply gravity
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clear_gravity_pass"),
                timestamp_writes: None,
            });
            self.clear_gravity_pipeline
                .dispatch(&mut pass, &clear_gravity_bg, self.particle_count);
        }

        // 2. Density (SPH kernel summation)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("density_pass"),
                timestamp_writes: None,
            });
            self.density_pipeline
                .dispatch(&mut pass, &density_particle_bg, &density_grid_bg, self.particle_count);
        }

        // 3. Pressure (Tait equation)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pressure_pass"),
                timestamp_writes: None,
            });
            self.pressure_pipeline
                .dispatch(&mut pass, &pressure_bg, self.particle_count);
        }

        // 4. Forces (pressure gradient + viscosity)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forces_pass"),
                timestamp_writes: None,
            });
            self.forces_pipeline
                .dispatch(&mut pass, &forces_particle_bg, &forces_grid_bg, self.particle_count);
        }

        // 5. Integration + boundary + obstacles
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("integrate_pass"),
                timestamp_writes: None,
            });
            self.integrate_pipeline
                .dispatch(&mut pass, &integrate_bg, self.particle_count);
        }

        queue.submit(std::iter::once(encoder.finish()));

        self.time += self.config.time_step;
        self.step_count += 1;
    }

    /// Download particle positions and velocities for CPU-side rendering.
    pub fn download_for_rendering(&self) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
        if self.particle_count == 0 {
            return (Vec::new(), Vec::new());
        }
        let positions = self.buffers.download_positions(
            &self.context.device,
            &self.context.queue,
            self.particle_count,
        );
        let velocities = self.buffers.download_velocities(
            &self.context.device,
            &self.context.queue,
            self.particle_count,
        );
        (positions, velocities)
    }

    /// Load a scene, spawning particles from all emitters.
    pub fn load_scene(&mut self, scene_desc: &SceneDescription) {
        self.config = scene_desc.config.clone();
        self.cpu_particles.clear();
        self.obstacles.clear();
        self.time = 0.0;
        self.step_count = 0;

        let spacing = scene_desc.emitters.first().map_or(0.02, |e| e.spacing);
        let mass = if self.config.particle_mass > 0.0 {
            self.config.particle_mass
        } else {
            self.config.compute_particle_mass(spacing, Dim2::DIM)
        };

        for emitter in &scene_desc.emitters {
            scene::spawn_from_emitter::<Dim2>(emitter, &mut self.cpu_particles, mass);
        }

        self.dirty = true;
        log::info!(
            "GPU: Loaded scene '{}' with {} particles",
            scene_desc.name,
            self.cpu_particles.len()
        );
    }

    /// Reset the simulation.
    pub fn reset(&mut self) {
        self.cpu_particles.clear();
        self.obstacles.clear();
        self.time = 0.0;
        self.step_count = 0;
        self.particle_count = 0;
        self.dirty = false;
    }

    /// Spawn particles at a world position.
    pub fn spawn_particles_at(&mut self, x: f32, y: f32, radius: f32, count: usize) {
        // Download current GPU state if needed
        self.sync_from_gpu();

        let spacing = if count > 1 {
            (2.0 * radius) / (count as f32).sqrt()
        } else {
            0.0
        };

        let mass = if self.config.particle_mass > 0.0 {
            self.config.particle_mass
        } else {
            self.config.compute_particle_mass(0.02, Dim2::DIM)
        };

        let vel = Dim2::zero();
        let steps = (count as f32).sqrt().ceil() as i32;
        let mut spawned = 0;
        for ix in -steps..=steps {
            for iy in -steps..=steps {
                if spawned >= count {
                    self.dirty = true;
                    return;
                }
                let px = x + ix as f32 * spacing;
                let py = y + iy as f32 * spacing;
                let dx = px - x;
                let dy = py - y;
                if dx * dx + dy * dy <= radius * radius {
                    let pos = Dim2::from_slice(&[px, py]);
                    self.cpu_particles.add(pos, vel, mass);
                    spawned += 1;
                }
            }
        }
        self.dirty = true;
    }

    /// Erase particles within a radius.
    pub fn erase_particles_at(&mut self, x: f32, y: f32, radius: f32) {
        self.sync_from_gpu();

        let radius_sq = radius * radius;
        let mut i = 0;
        while i < self.cpu_particles.len() {
            let pos = &self.cpu_particles.positions[i];
            let dx = pos.x - x;
            let dy = pos.y - y;
            if dx * dx + dy * dy <= radius_sq {
                self.cpu_particles.remove(i);
            } else {
                i += 1;
            }
        }
        self.dirty = true;
    }

    /// Apply a force with distance-based falloff to particles near a position (for drag tool).
    pub fn apply_force_at(&mut self, x: f32, y: f32, fx: f32, fy: f32, radius: f32) {
        self.sync_from_gpu();

        let radius_sq = radius * radius;
        for i in 0..self.cpu_particles.len() {
            let pos = &self.cpu_particles.positions[i];
            let dx = pos.x - x;
            let dy = pos.y - y;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq < radius_sq {
                let factor = 1.0 - (dist_sq / radius_sq).sqrt();
                self.cpu_particles.velocities[i].x += fx * factor;
                self.cpu_particles.velocities[i].y += fy * factor;
            }
        }
        self.dirty = true;
    }

    /// Sync GPU state back to CPU particles (for mutation operations).
    fn sync_from_gpu(&mut self) {
        if self.dirty || self.particle_count == 0 {
            return;
        }

        let positions = self.buffers.download_positions(
            &self.context.device,
            &self.context.queue,
            self.particle_count,
        );
        let velocities = self.buffers.download_velocities(
            &self.context.device,
            &self.context.queue,
            self.particle_count,
        );
        let masses = self.buffers.download_masses(
            &self.context.device,
            &self.context.queue,
            self.particle_count,
        );

        let n = self.particle_count as usize;
        if positions.len() < n || velocities.len() < n {
            log::warn!("GPU readback returned incomplete data, skipping sync");
            return;
        }

        self.cpu_particles.clear();
        let fallback_mass = if self.config.particle_mass > 0.0 {
            self.config.particle_mass
        } else {
            self.config.compute_particle_mass(0.02, Dim2::DIM)
        };
        for i in 0..self.particle_count as usize {
            let pos = Dim2::from_slice(&positions[i]);
            let vel = Dim2::from_slice(&velocities[i]);
            let mass = masses.get(i).copied().unwrap_or(fallback_mass);
            self.cpu_particles.add(pos, vel, mass);
        }
    }

    /// Get a reference to the GPU device (for shared rendering).
    pub fn device(&self) -> &wgpu::Device {
        &self.context.device
    }

    /// Get a reference to the GPU queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.context.queue
    }

    pub fn is_gpu(&self) -> bool {
        true
    }

    pub fn backend_name(&self) -> &str {
        "GPU"
    }
}
