use common::{Dim2, Dimension, SimConfig};
use sim_core::obstacle::Obstacle;
use sim_core::scene::SceneDescription;
use sim_core::Simulation;
use sim_gpu::GpuSimulation;

/// Unified backend that delegates to either a CPU or GPU simulation.
pub enum SimulationBackend {
    Cpu(Simulation<Dim2>),
    Gpu(GpuSimulation),
}

impl SimulationBackend {
    pub fn new_cpu(config: SimConfig) -> Self {
        SimulationBackend::Cpu(Simulation::new(config))
    }

    pub fn new_gpu(config: SimConfig) -> Option<Self> {
        GpuSimulation::create(config).map(SimulationBackend::Gpu)
    }

    pub fn step(&mut self) {
        match self {
            SimulationBackend::Cpu(sim) => sim.step(),
            SimulationBackend::Gpu(gpu) => gpu.step(),
        }
    }

    pub fn particle_count(&self) -> usize {
        match self {
            SimulationBackend::Cpu(sim) => sim.particles.len(),
            SimulationBackend::Gpu(gpu) => gpu.particle_count(),
        }
    }

    pub fn time(&self) -> f32 {
        match self {
            SimulationBackend::Cpu(sim) => sim.time,
            SimulationBackend::Gpu(gpu) => gpu.time,
        }
    }

    pub fn step_count(&self) -> u64 {
        match self {
            SimulationBackend::Cpu(sim) => sim.step_count,
            SimulationBackend::Gpu(gpu) => gpu.step_count,
        }
    }

    pub fn config(&self) -> &SimConfig {
        match self {
            SimulationBackend::Cpu(sim) => &sim.config,
            SimulationBackend::Gpu(gpu) => &gpu.config,
        }
    }

    pub fn config_mut(&mut self) -> &mut SimConfig {
        match self {
            SimulationBackend::Cpu(sim) => &mut sim.config,
            SimulationBackend::Gpu(gpu) => &mut gpu.config,
        }
    }

    pub fn obstacles(&self) -> &[Obstacle<Dim2>] {
        match self {
            SimulationBackend::Cpu(sim) => &sim.obstacles,
            SimulationBackend::Gpu(gpu) => &gpu.obstacles,
        }
    }

    pub fn push_obstacle(&mut self, obstacle: Obstacle<Dim2>) {
        match self {
            SimulationBackend::Cpu(sim) => sim.obstacles.push(obstacle),
            SimulationBackend::Gpu(gpu) => gpu.obstacles.push(obstacle),
        }
    }

    pub fn remove_obstacle(&mut self, index: usize) {
        match self {
            SimulationBackend::Cpu(sim) => {
                sim.obstacles.swap_remove(index);
            }
            SimulationBackend::Gpu(gpu) => {
                gpu.obstacles.swap_remove(index);
            }
        }
    }

    pub fn load_scene(&mut self, scene: &SceneDescription) {
        match self {
            SimulationBackend::Cpu(sim) => sim.load_scene(scene),
            SimulationBackend::Gpu(gpu) => gpu.load_scene(scene),
        }
    }

    pub fn reset(&mut self) {
        match self {
            SimulationBackend::Cpu(sim) => sim.reset(),
            SimulationBackend::Gpu(gpu) => gpu.reset(),
        }
    }

    pub fn spawn_particles_at(&mut self, x: f32, y: f32, radius: f32, count: usize) {
        match self {
            SimulationBackend::Cpu(sim) => sim.spawn_particles_at(x, y, radius, count),
            SimulationBackend::Gpu(gpu) => gpu.spawn_particles_at(x, y, radius, count),
        }
    }

    pub fn erase_particles_at(&mut self, x: f32, y: f32, radius: f32) {
        match self {
            SimulationBackend::Cpu(sim) => sim.erase_particles_at(x, y, radius),
            SimulationBackend::Gpu(gpu) => gpu.erase_particles_at(x, y, radius),
        }
    }

    /// Apply a force with distance-based falloff to particles near a position.
    pub fn apply_force_at(&mut self, x: f32, y: f32, fx: f32, fy: f32, radius: f32) {
        match self {
            SimulationBackend::Cpu(sim) => {
                let radius_sq = radius * radius;
                for i in 0..sim.particles.len() {
                    let pos = &sim.particles.positions[i];
                    let px = Dim2::component(pos, 0);
                    let py = Dim2::component(pos, 1);
                    let dx = px - x;
                    let dy = py - y;
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq < radius_sq {
                        let factor = 1.0 - (dist_sq / radius_sq).sqrt();
                        let force = Dim2::from_slice(&[fx * factor, fy * factor]);
                        sim.particles.velocities[i] += force;
                    }
                }
            }
            SimulationBackend::Gpu(gpu) => {
                gpu.apply_force_at(x, y, fx, fy, radius);
            }
        }
    }

    /// Get particle positions and velocities as `[f32; 2]` arrays for rendering.
    pub fn particle_data_for_rendering(&self) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
        match self {
            SimulationBackend::Cpu(sim) => {
                let n = sim.particles.len();
                let mut positions = Vec::with_capacity(n);
                let mut velocities = Vec::with_capacity(n);
                for i in 0..n {
                    let pos = &sim.particles.positions[i];
                    let vel = &sim.particles.velocities[i];
                    positions.push([Dim2::component(pos, 0), Dim2::component(pos, 1)]);
                    velocities.push([Dim2::component(vel, 0), Dim2::component(vel, 1)]);
                }
                (positions, velocities)
            }
            SimulationBackend::Gpu(gpu) => gpu.download_for_rendering(),
        }
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, SimulationBackend::Gpu(_))
    }

    pub fn backend_name(&self) -> &str {
        match self {
            SimulationBackend::Cpu(_) => "CPU",
            SimulationBackend::Gpu(_) => "GPU",
        }
    }
}
