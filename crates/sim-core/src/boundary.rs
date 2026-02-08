use common::{Dimension, SimConfig};

use crate::particle::ParticleStorage;

/// Enforce domain boundaries by clamping positions and reflecting velocities.
pub fn enforce_boundaries<D: Dimension>(particles: &mut ParticleStorage<D>, config: &SimConfig) {
    let domain_min = D::from_slice(&config.domain_min);
    let domain_max = D::from_slice(&config.domain_max);
    let damping = config.boundary_damping;

    for i in 0..particles.len() {
        for d in 0..D::DIM {
            let c = D::component(&particles.positions[i], d);
            let min_c = D::component(&domain_min, d);
            let max_c = D::component(&domain_max, d);

            if c < min_c {
                D::set_component(&mut particles.positions[i], d, min_c);
                let v = D::component(&particles.velocities[i], d);
                D::set_component(&mut particles.velocities[i], d, -v * damping);
            } else if c > max_c {
                D::set_component(&mut particles.positions[i], d, max_c);
                let v = D::component(&particles.velocities[i], d);
                D::set_component(&mut particles.velocities[i], d, -v * damping);
            }
        }
    }
}
