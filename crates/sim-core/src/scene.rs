use serde::{Deserialize, Serialize};

use common::{Dimension, SimConfig};

use crate::particle::ParticleStorage;

/// A particle emitter that spawns particles in a rectangular region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Emitter {
    /// Center position [x, y].
    pub center: [f32; 2],
    /// Half-extent of the emitter region [width/2, height/2].
    pub half_size: [f32; 2],
    /// Particle spacing within the emitter.
    pub spacing: f32,
    /// Initial velocity of emitted particles [vx, vy].
    pub initial_velocity: [f32; 2],
}

/// Scene description: a collection of emitters and simulation config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneDescription {
    pub name: String,
    pub config: SimConfig,
    pub emitters: Vec<Emitter>,
}

impl SceneDescription {
    /// Create a default dam break scene.
    pub fn dam_break() -> Self {
        Self {
            name: "Dam Break".to_string(),
            config: SimConfig::default(),
            emitters: vec![Emitter {
                center: [0.25, 0.5],
                half_size: [0.2, 0.4],
                spacing: 0.02,
                initial_velocity: [0.0, 0.0],
            }],
        }
    }

    /// Create a default double emitter scene.
    pub fn double_emitter() -> Self {
        Self {
            name: "Double Emitter".to_string(),
            config: SimConfig::default(),
            emitters: vec![
                Emitter {
                    center: [0.2, 0.8],
                    half_size: [0.1, 0.1],
                    spacing: 0.02,
                    initial_velocity: [2.0, -1.0],
                },
                Emitter {
                    center: [0.8, 0.8],
                    half_size: [0.1, 0.1],
                    spacing: 0.02,
                    initial_velocity: [-2.0, -1.0],
                },
            ],
        }
    }
}

/// Spawn particles from an emitter into storage.
pub fn spawn_from_emitter<D: Dimension>(
    emitter: &Emitter,
    particles: &mut ParticleStorage<D>,
    particle_mass: f32,
) {
    let min_x = emitter.center[0] - emitter.half_size[0];
    let max_x = emitter.center[0] + emitter.half_size[0];
    let min_y = emitter.center[1] - emitter.half_size[1];
    let max_y = emitter.center[1] + emitter.half_size[1];
    let spacing = emitter.spacing;

    let vel = D::from_slice(&emitter.initial_velocity);

    let mut x = min_x;
    while x <= max_x {
        let mut y = min_y;
        while y <= max_y {
            let pos = D::from_slice(&[x, y]);
            particles.add(pos, vel, particle_mass);
            y += spacing;
        }
        x += spacing;
    }
}
