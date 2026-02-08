use rayon::prelude::*;

use common::{Dimension, SimConfig};

use crate::neighbor::SpatialHashGrid;
use crate::particle::ParticleStorage;
use crate::sph::kernel::{KernelParams, SmoothingKernel};

/// Compute artificial viscosity forces for all particles.
/// f_viscosity_i = mu * sum_j m_j * (v_j - v_i) / rho_j * laplacian_W(|r_i - r_j|, h)
pub fn compute_viscosity_forces<D: Dimension>(
    particles: &mut ParticleStorage<D>,
    grid: &SpatialHashGrid<D>,
    kernel: &(dyn SmoothingKernel<D> + Sync),
    config: &SimConfig,
    params: &KernelParams,
) {
    let mu = config.viscosity;
    let support_sq = params.support_sq;
    let positions = &particles.positions;
    let velocities = &particles.velocities;
    let densities = &particles.densities;
    let masses = &particles.masses;

    // Parallel viscosity force computation.
    let forces: Vec<D::Vector> = (0..positions.len())
        .into_par_iter()
        .map(|i| {
            let mut force = D::zero();
            let pos_i = positions[i];

            for j in grid.query_neighbors_iter(&pos_i) {
                if i == j {
                    continue;
                }

                let r_vec = pos_i - positions[j];
                let dist_sq = D::magnitude_sq(&r_vec);

                // Early rejection: skip particles outside kernel support.
                if dist_sq >= support_sq {
                    continue;
                }

                let r = dist_sq.sqrt();
                if r < 1e-10 {
                    continue;
                }

                let rho_j = densities[j].max(1e-6);
                let laplacian = kernel.laplacian_w(r, params);
                let vel_diff = velocities[j] - velocities[i];

                force += vel_diff * (mu * masses[j] * laplacian / rho_j);
            }

            force
        })
        .collect();

    for (accel, force) in particles.accelerations.iter_mut().zip(&forces) {
        *accel += *force;
    }
}
