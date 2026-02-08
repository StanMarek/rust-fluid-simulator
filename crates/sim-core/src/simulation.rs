use common::{Dimension, SimConfig};

use crate::boundary;
use crate::particle::ParticleStorage;
use crate::scene::{self, SceneDescription};
use crate::solver;

/// Top-level simulation state.
/// Phase 1: gravity-only solver with boundary reflection.
/// Phase 2 will add SPH (neighbor search, density, pressure, viscosity).
pub struct Simulation<D: Dimension> {
    pub particles: ParticleStorage<D>,
    pub config: SimConfig,
    pub time: f32,
    pub step_count: u64,
}

impl<D: Dimension> Simulation<D> {
    /// Create a new empty simulation.
    pub fn new(config: SimConfig) -> Self {
        Self {
            particles: ParticleStorage::new(),
            config,
            time: 0.0,
            step_count: 0,
        }
    }

    /// Load a scene, spawning particles from all emitters.
    pub fn load_scene(&mut self, scene: &SceneDescription) {
        self.config = scene.config.clone();
        self.particles.clear();
        self.time = 0.0;
        self.step_count = 0;

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

    /// Advance the simulation by one time step.
    /// Phase 1: gravity + boundary enforcement only.
    pub fn step(&mut self) {
        let dt = self.config.time_step;

        // 1. Clear accelerations
        self.particles.clear_accelerations();

        // 2. Apply gravity
        solver::apply_gravity::<D>(&mut self.particles, &self.config);

        // 3. Integrate (symplectic Euler)
        solver::integrate_symplectic_euler::<D>(&mut self.particles, &self.config, dt);

        // 4. Enforce boundaries
        boundary::enforce_boundaries::<D>(&mut self.particles, &self.config);

        self.time += dt;
        self.step_count += 1;
    }

    /// Reset the simulation (clear particles and time).
    pub fn reset(&mut self) {
        self.particles.clear();
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
