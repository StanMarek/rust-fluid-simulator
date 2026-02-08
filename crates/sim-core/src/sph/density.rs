use rayon::prelude::*;

use common::Dimension;

use crate::neighbor::SpatialHashGrid;
use crate::particle::ParticleStorage;
use crate::sph::kernel::{KernelParams, SmoothingKernel};

/// Compute density for all particles using SPH summation.
/// rho_i = sum_j m_j * W(|r_i - r_j|, h)
pub fn compute_densities<D: Dimension>(
    particles: &mut ParticleStorage<D>,
    grid: &SpatialHashGrid<D>,
    kernel: &(dyn SmoothingKernel<D> + Sync),
    params: &KernelParams,
) {
    let support_sq = params.support_sq;
    let positions = &particles.positions;
    let masses = &particles.masses;

    // Parallel density computation: each particle writes only to its own density.
    let new_densities: Vec<f32> = (0..positions.len())
        .into_par_iter()
        .map(|i| {
            let mut density = 0.0;
            let pos_i = positions[i];

            for j in grid.query_neighbors_iter(&pos_i) {
                let diff = pos_i - positions[j];
                let dist_sq = D::magnitude_sq(&diff);

                // Early rejection: skip particles outside kernel support.
                if dist_sq >= support_sq {
                    continue;
                }

                let r = dist_sq.sqrt();
                density += masses[j] * kernel.w(r, params);
            }

            density
        })
        .collect();

    // Note: self-contribution (W(0,h)) is included via the grid
    // returning particle i in its own cell.

    particles.densities = new_densities;
}
