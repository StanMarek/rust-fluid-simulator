use serde::{Deserialize, Serialize};

/// Simulation configuration. All parameters are exposed via UI sliders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    /// Smoothing radius (h). Determines kernel support and neighbor search radius.
    pub smoothing_radius: f32,
    /// Rest density (ρ₀). For water, typically 1000.0.
    pub rest_density: f32,
    /// Stiffness coefficient (k) for the equation of state.
    pub stiffness: f32,
    /// Viscosity coefficient (μ).
    pub viscosity: f32,
    /// Gravity vector [x, y]. For 2D, typically [0.0, -9.81].
    pub gravity: [f32; 2],
    /// Fixed simulation time step (dt).
    pub time_step: f32,
    /// Particle mass. Auto-computed from spacing and rest density if zero.
    pub particle_mass: f32,
    /// Domain minimum bounds [x, y].
    pub domain_min: [f32; 2],
    /// Domain maximum bounds [x, y].
    pub domain_max: [f32; 2],
    /// Velocity damping factor when hitting boundaries (0 = full absorption, 1 = full bounce).
    pub boundary_damping: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            smoothing_radius: 0.1,
            rest_density: 1000.0,
            stiffness: 1000.0,
            viscosity: 0.1,
            gravity: [0.0, -9.81],
            time_step: 0.001,
            particle_mass: 0.0, // Auto-compute
            domain_min: [0.0, 0.0],
            domain_max: [1.0, 1.0],
            boundary_damping: 0.5,
        }
    }
}

impl SimConfig {
    /// Compute particle mass from spacing and rest density.
    /// mass = rest_density * spacing^DIM
    pub fn compute_particle_mass(&self, spacing: f32, dim: usize) -> f32 {
        self.rest_density * spacing.powi(dim as i32)
    }
}
