use common::{Dimension, SimConfig};

use crate::neighbor::SpatialHashGrid;
use crate::particle::ParticleStorage;
use crate::sph::kernel::SmoothingKernel;

/// Compute pressure from density using the Tait equation of state (WCSPH).
/// p = k * ((ρ/ρ₀)^γ - 1)
/// Using γ = 7 for water.
pub fn compute_pressures<D: Dimension>(particles: &mut ParticleStorage<D>, config: &SimConfig) {
    let gamma = 7.0_f32;
    for i in 0..particles.len() {
        let rho_ratio = particles.densities[i] / config.rest_density;
        particles.pressures[i] = config.stiffness * (rho_ratio.powf(gamma) - 1.0);
    }
}

/// Compute pressure gradient forces for all particles.
/// f_pressure_i = -Σ_j m_j * (p_i/ρ_i² + p_j/ρ_j²) * ∇W(r_i - r_j, h)
pub fn compute_pressure_forces<D: Dimension>(
    particles: &mut ParticleStorage<D>,
    grid: &SpatialHashGrid<D>,
    kernel: &dyn SmoothingKernel<D>,
    h: f32,
) {
    let n = particles.len();

    // Collect pressure force contributions (avoid borrow conflict)
    let mut forces: Vec<D::Vector> = vec![D::zero(); n];

    for i in 0..n {
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

            let grad = kernel.grad_w(r_vec, r, h);
            let rho_i = particles.densities[i].max(1e-6);
            let rho_j = particles.densities[j].max(1e-6);
            let pressure_term =
                particles.pressures[i] / (rho_i * rho_i) + particles.pressures[j] / (rho_j * rho_j);

            let force = grad * (-particles.masses[j] * pressure_term);
            forces[i] += force;
        }
    }

    // Apply forces as accelerations
    for i in 0..n {
        particles.accelerations[i] += forces[i];
    }
}
