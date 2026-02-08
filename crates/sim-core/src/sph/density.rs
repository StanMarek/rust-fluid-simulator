use common::Dimension;

use crate::neighbor::SpatialHashGrid;
use crate::particle::ParticleStorage;
use crate::sph::kernel::SmoothingKernel;

/// Compute density for all particles using SPH summation.
/// ρ_i = Σ_j m_j * W(|r_i - r_j|, h)
pub fn compute_densities<D: Dimension>(
    particles: &mut ParticleStorage<D>,
    grid: &SpatialHashGrid<D>,
    kernel: &dyn SmoothingKernel<D>,
    h: f32,
) {
    let n = particles.len();
    for i in 0..n {
        let mut density = 0.0;
        let neighbors = grid.query_neighbors(&particles.positions[i], h);

        for &j in &neighbors {
            let diff = particles.positions[i] - particles.positions[j];
            let r = D::magnitude(&diff);
            density += particles.masses[j] * kernel.w(r, h);
        }

        // Note: self-contribution (W(0,h)) is included via the grid
        // returning particle i in its own cell.

        particles.densities[i] = density;
    }
}
