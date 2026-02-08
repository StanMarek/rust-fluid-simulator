/// Events for communication between the simulation and the UI.
#[derive(Debug, Clone)]
pub enum SimEvent {
    /// Start or resume the simulation.
    Play,
    /// Pause the simulation.
    Pause,
    /// Advance the simulation by a single time step.
    Step,
    /// Reset the simulation to the initial state.
    Reset,
    /// Spawn particles at the given world position with the given radius.
    SpawnParticles {
        x: f32,
        y: f32,
        radius: f32,
        count: usize,
    },
    /// Apply an external force at the given world position.
    ApplyForce { x: f32, y: f32, fx: f32, fy: f32 },
    /// Erase particles within the given radius of the world position.
    EraseParticles { x: f32, y: f32, radius: f32 },
    /// Update simulation configuration.
    UpdateConfig,
}

/// Current state of the simulation for UI display.
#[derive(Debug, Clone)]
pub struct SimStatus {
    pub particle_count: usize,
    pub sim_time: f32,
    pub step_count: u64,
    pub is_running: bool,
}

impl Default for SimStatus {
    fn default() -> Self {
        Self {
            particle_count: 0,
            sim_time: 0.0,
            step_count: 0,
            is_running: false,
        }
    }
}
