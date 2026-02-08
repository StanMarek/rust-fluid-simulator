use common::{Dimension, SimConfig};

use crate::neighbor::SpatialHashGrid;
use crate::particle::ParticleStorage;
use crate::sph::kernel::SmoothingKernel;

/// Compute artificial viscosity forces for all particles.
/// f_viscosity_i = μ * Σ_j m_j * (v_j - v_i) / ρ_j * ∇²W(|r_i - r_j|, h)
pub fn compute_viscosity_forces<D: Dimension>(
    particles: &mut ParticleStorage<D>,
    grid: &SpatialHashGrid<D>,
    kernel: &dyn SmoothingKernel<D>,
    config: &SimConfig,
) {
    let n = particles.len();
    let h = config.smoothing_radius;
    let mu = config.viscosity;

    let mut forces: Vec<D::Vector> = vec![D::zero(); n];

    for (i, force) in forces.iter_mut().enumerate() {
        let neighbors = grid.query_neighbors(&particles.positions[i], h);

        for &j in &neighbors {
            if i == j {
                continue;
            }

            let r_vec = particles.positions[i] - particles.positions[j];
            let r = D::magnitude(&r_vec);

            if r < 1e-10 {
                continue;
            }

            let rho_j = particles.densities[j].max(1e-6);
            let laplacian = kernel.laplacian_w(r, h);
            let vel_diff = particles.velocities[j] - particles.velocities[i];

            *force += vel_diff * (mu * particles.masses[j] * laplacian / rho_j);
        }
    }

    for (accel, force) in particles.accelerations.iter_mut().zip(&forces) {
        *accel += *force;
    }
}
