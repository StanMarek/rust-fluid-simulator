use common::{Dimension, SimConfig};

use crate::obstacle::Obstacle;
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

/// Enforce obstacle boundaries by pushing particles out and reflecting velocities.
pub fn enforce_obstacle_boundaries<D: Dimension>(
    particles: &mut ParticleStorage<D>,
    obstacles: &[Obstacle<D>],
    damping: f32,
) {
    for i in 0..particles.len() {
        for obstacle in obstacles {
            let dist = obstacle.sdf(&particles.positions[i]);
            if dist < 0.0 {
                let normal = obstacle.normal(&particles.positions[i]);
                // Push particle to the surface
                particles.positions[i] += normal * (-dist + 1e-4);

                // Reflect velocity component along normal
                let v_dot_n = D::dot(&particles.velocities[i], &normal);
                if v_dot_n < 0.0 {
                    particles.velocities[i] += normal * (-v_dot_n * (1.0 + damping));
                }
            }
        }
    }
}
