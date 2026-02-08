use common::{Dimension, SimConfig};

use crate::particle::ParticleStorage;

/// Time integration methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationMethod {
    SymplecticEuler,
    Leapfrog,
}

/// Perform symplectic Euler integration.
/// v(t+dt) = v(t) + a(t) * dt
/// x(t+dt) = x(t) + v(t+dt) * dt
pub fn integrate_symplectic_euler<D: Dimension>(
    particles: &mut ParticleStorage<D>,
    _config: &SimConfig,
    dt: f32,
) {
    for i in 0..particles.len() {
        // Update velocity
        particles.velocities[i] += particles.accelerations[i] * dt;
        // Update position with new velocity
        particles.positions[i] += particles.velocities[i] * dt;
    }
}

/// Apply gravity to all particle accelerations.
pub fn apply_gravity<D: Dimension>(particles: &mut ParticleStorage<D>, config: &SimConfig) {
    let gravity = D::from_slice(&config.gravity);
    for i in 0..particles.len() {
        particles.accelerations[i] += gravity;
    }
}
