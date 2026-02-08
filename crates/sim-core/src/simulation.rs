use common::{Dimension, SimConfig};

use crate::boundary;
use crate::neighbor::SpatialHashGrid;
use crate::obstacle::Obstacle;
use crate::particle::ParticleStorage;
use crate::scene::{self, SceneDescription};
use crate::solver;
use crate::sph::density;
use crate::sph::kernel::{CubicSplineKernel, KernelParams, SmoothingKernel};
use crate::sph::pressure;
use crate::sph::viscosity;

/// Top-level simulation state.
/// Holds particle data, SPH neighbor grid, and kernel for fluid simulation.
pub struct Simulation<D: Dimension> {
    pub particles: ParticleStorage<D>,
    pub config: SimConfig,
    pub time: f32,
    pub step_count: u64,
    pub obstacles: Vec<Obstacle<D>>,
    /// Spatial hash grid for neighbor search, rebuilt each step.
    grid: SpatialHashGrid<D>,
    /// SPH smoothing kernel.
    kernel: CubicSplineKernel,
}

impl<D: Dimension> Simulation<D> {
    /// Create a new empty simulation.
    pub fn new(config: SimConfig) -> Self {
        // Cell size = 2h (kernel support radius) so 3x3 query covers all neighbors.
        let grid = SpatialHashGrid::new(2.0 * config.smoothing_radius);
        Self {
            particles: ParticleStorage::new(),
            config,
            time: 0.0,
            step_count: 0,
            obstacles: Vec::new(),
            grid,
            kernel: CubicSplineKernel,
        }
    }

    /// Load a scene, spawning particles from all emitters.
    pub fn load_scene(&mut self, scene: &SceneDescription) {
        self.config = scene.config.clone();
        self.particles.clear();
        self.obstacles.clear();
        self.time = 0.0;
        self.step_count = 0;

        // Sync grid cell size with new config (cell_size = 2h for full kernel support).
        self.grid.set_cell_size(2.0 * self.config.smoothing_radius);

        let spacing = scene.emitters.first().map_or(0.02, |e| e.spacing);
        let mass = if self.config.particle_mass > 0.0 {
            self.config.particle_mass
        } else {
            self.config.compute_particle_mass(spacing, D::DIM)
        };

        for emitter in &scene.emitters {
            scene::spawn_from_emitter::<D>(emitter, &mut self.particles, mass);
        }

        log::info!(
            "Loaded scene '{}' with {} particles",
            scene.name,
            self.particles.len()
        );
    }

    /// Reset the simulation (clear particles, obstacles, and time).
    pub fn reset(&mut self) {
        self.particles.clear();
        self.obstacles.clear();
        self.time = 0.0;
        self.step_count = 0;
    }

    /// Spawn particles in a circular region at the given world position.
    pub fn spawn_particles_at(&mut self, x: f32, y: f32, radius: f32, count: usize) {
        let spacing = if count > 1 {
            (2.0 * radius) / (count as f32).sqrt()
        } else {
            0.0
        };

        let mass = if self.config.particle_mass > 0.0 {
            self.config.particle_mass
        } else {
            self.config.compute_particle_mass(0.02, D::DIM)
        };

        let vel = D::zero();

        // Spawn in a grid pattern within the circle
        let steps = (count as f32).sqrt().ceil() as i32;
        let mut spawned = 0;
        for ix in -steps..=steps {
            for iy in -steps..=steps {
                if spawned >= count {
                    return;
                }
                let px = x + ix as f32 * spacing;
                let py = y + iy as f32 * spacing;
                let dx = px - x;
                let dy = py - y;
                if dx * dx + dy * dy <= radius * radius {
                    let pos = D::from_slice(&[px, py]);
                    self.particles.add(pos, vel, mass);
                    spawned += 1;
                }
            }
        }
    }

    /// Remove particles within a radius of the given world position.
    pub fn erase_particles_at(&mut self, x: f32, y: f32, radius: f32) {
        let radius_sq = radius * radius;
        let mut i = 0;
        while i < self.particles.len() {
            let pos = &self.particles.positions[i];
            let px = D::component(pos, 0);
            let py = D::component(pos, 1);
            let dx = px - x;
            let dy = py - y;
            if dx * dx + dy * dy <= radius_sq {
                self.particles.remove(i);
                // Don't increment i since swap_remove puts a new element at i
            } else {
                i += 1;
            }
        }
    }
}

impl<D: Dimension> Simulation<D>
where
    CubicSplineKernel: SmoothingKernel<D>,
{
    /// Advance the simulation by one time step.
    /// Full SPH loop: neighbor search, density, pressure, forces, integration.
    pub fn step(&mut self) {
        let dt = self.config.time_step;
        let h = self.config.smoothing_radius;
        let params = KernelParams::new(h);

        // 1. Clear accelerations
        self.particles.clear_accelerations();

        // 2. Build neighbor grid (cell_size = 2h for full kernel support)
        self.grid.set_cell_size(2.0 * h);
        self.grid.build(&self.particles.positions);

        // 3. Compute densities (SPH summation)
        density::compute_densities(&mut self.particles, &self.grid, &self.kernel, &params);

        // 4. Compute pressures (Tait equation of state)
        pressure::compute_pressures::<D>(&mut self.particles, &self.config);

        // 5. Compute pressure gradient forces
        pressure::compute_pressure_forces(&mut self.particles, &self.grid, &self.kernel, &params);

        // 6. Compute viscosity forces
        viscosity::compute_viscosity_forces(
            &mut self.particles,
            &self.grid,
            &self.kernel,
            &self.config,
            &params,
        );

        // 7. Apply gravity
        solver::apply_gravity::<D>(&mut self.particles, &self.config);

        // 8. Integrate (symplectic Euler)
        solver::integrate_symplectic_euler::<D>(&mut self.particles, &self.config, dt);

        // 9. Enforce boundaries
        boundary::enforce_boundaries::<D>(&mut self.particles, &self.config);

        // 10. Enforce obstacle boundaries
        boundary::enforce_obstacle_boundaries::<D>(
            &mut self.particles,
            &self.obstacles,
            self.config.boundary_damping,
        );

        self.time += dt;
        self.step_count += 1;
    }
}
