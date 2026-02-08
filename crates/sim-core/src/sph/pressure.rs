use rayon::prelude::*;

use common::{Dimension, SimConfig};

use crate::neighbor::SpatialHashGrid;
use crate::particle::ParticleStorage;
use crate::sph::kernel::{KernelParams, SmoothingKernel};

/// Compute pressure from density using the Tait equation of state (WCSPH).
/// p = k * ((rho/rho0)^7 - 1), clamped to non-negative for free-surface stability.
pub fn compute_pressures<D: Dimension>(particles: &mut ParticleStorage<D>, config: &SimConfig) {
    let stiffness = config.stiffness;
    let rest_density = config.rest_density;

    particles
        .pressures
        .par_iter_mut()
        .zip(particles.densities.par_iter())
        .for_each(|(pressure, &density)| {
            let rho_ratio = density / rest_density;
            // Integer power: rho_ratio^7 = (rho_ratio^2)^2 * rho_ratio^2 * rho_ratio
            let r2 = rho_ratio * rho_ratio;
            let r4 = r2 * r2;
            let r7 = r4 * r2 * rho_ratio;
            // Clamp to >= 0: negative pressure (tensile instability) causes
            // particle clumping and explosions at free surfaces.
            *pressure = (stiffness * (r7 - 1.0)).max(0.0);
        });
}

/// Compute pressure gradient forces for all particles.
/// f_pressure_i = -sum_j m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_W(r_i - r_j, h)
pub fn compute_pressure_forces<D: Dimension>(
    particles: &mut ParticleStorage<D>,
    grid: &SpatialHashGrid<D>,
    kernel: &(dyn SmoothingKernel<D> + Sync),
    params: &KernelParams,
) {
    let support_sq = params.support_sq;
    let positions = &particles.positions;
    let densities = &particles.densities;
    let pressures = &particles.pressures;
    let masses = &particles.masses;

    // Parallel pressure force computation.
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

                let grad = kernel.grad_w(r_vec, r, params);
                let rho_i = densities[i].max(1e-6);
                let rho_j = densities[j].max(1e-6);
                let pressure_term = pressures[i] / (rho_i * rho_i) + pressures[j] / (rho_j * rho_j);

                force += grad * (-masses[j] * pressure_term);
            }

            force
        })
        .collect();

    // Apply forces as accelerations.
    for (accel, force) in particles.accelerations.iter_mut().zip(&forces) {
        *accel += *force;
    }
}
