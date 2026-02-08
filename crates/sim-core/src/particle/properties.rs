//! Particle property indices and documentation.
//!
//! Particles are stored in SoA (Struct of Arrays) layout for cache-friendly
//! access and GPU buffer mapping. See `ParticleStorage`.
//!
//! Each particle has:
//! - **position**: Current world-space position
//! - **velocity**: Current velocity vector
//! - **acceleration**: Accumulated forces / mass for this timestep
//! - **density**: Computed SPH density (ρ)
//! - **pressure**: Computed from equation of state
//! - **mass**: Particle mass (typically uniform)

/// Default particle radius for rendering.
pub const DEFAULT_PARTICLE_RADIUS: f32 = 0.005;

/// Minimum distance between particles during initialization.
pub const DEFAULT_PARTICLE_SPACING: f32 = 0.02;
