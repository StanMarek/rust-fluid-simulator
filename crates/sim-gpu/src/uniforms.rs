use bytemuck::{Pod, Zeroable};
use common::SimConfig;
use sim_core::obstacle::Obstacle;
use common::Dim2;

/// GPU-side simulation parameters uniform buffer.
/// Layout: 5 x vec4 = 80 bytes, all 16-byte aligned.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct SimParamsUniform {
    // Row 0: kernel params
    pub h: f32,
    pub h_sq: f32,
    pub h_inv: f32,
    pub sigma: f32,

    // Row 1: SPH params
    pub support_sq: f32,
    pub rest_density: f32,
    pub stiffness: f32,
    pub viscosity: f32,

    // Row 2: integration params
    pub gravity_x: f32,
    pub gravity_y: f32,
    pub dt: f32,
    pub boundary_damping: f32,

    // Row 3: domain bounds
    pub domain_min_x: f32,
    pub domain_min_y: f32,
    pub domain_max_x: f32,
    pub domain_max_y: f32,

    // Row 4: grid + particle info
    pub particle_count: u32,
    pub grid_table_size: u32,
    pub grid_cell_size: f32,
    pub _pad: f32,
}

impl SimParamsUniform {
    pub fn from_config(config: &SimConfig, particle_count: u32) -> Self {
        let h = config.smoothing_radius;
        let sigma = 10.0 / (7.0 * std::f32::consts::PI * h * h);
        Self {
            h,
            h_sq: h * h,
            h_inv: 1.0 / h,
            sigma,
            support_sq: 4.0 * h * h,
            rest_density: config.rest_density,
            stiffness: config.stiffness,
            viscosity: config.viscosity,
            gravity_x: config.gravity[0],
            gravity_y: config.gravity[1],
            dt: config.time_step,
            boundary_damping: config.boundary_damping,
            domain_min_x: config.domain_min[0],
            domain_min_y: config.domain_min[1],
            domain_max_x: config.domain_max[0],
            domain_max_y: config.domain_max[1],
            particle_count,
            grid_table_size: common::GRID_TABLE_SIZE,
            grid_cell_size: 2.0 * h,
            _pad: 0.0,
        }
    }
}

const _: () = assert!(std::mem::size_of::<SimParamsUniform>() == 80);

/// GPU-side obstacle data. Matches WGSL ObstacleData layout.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct ObstacleGpu {
    /// 0 = Circle, 1 = Box
    pub obstacle_type: u32,
    pub _pad: [f32; 3],
    /// Circle: [cx, cy, r, 0], Box: [min_x, min_y, max_x, max_y]
    pub data: [f32; 4],
}

impl ObstacleGpu {
    pub fn from_obstacle(obstacle: &Obstacle<Dim2>) -> Self {
        match obstacle {
            Obstacle::Circle { center, radius } => Self {
                obstacle_type: 0,
                _pad: [0.0; 3],
                data: [center.x, center.y, *radius, 0.0],
            },
            Obstacle::Box { min, max } => Self {
                obstacle_type: 1,
                _pad: [0.0; 3],
                data: [min.x, min.y, max.x, max.y],
            },
        }
    }
}

/// GPU-side obstacle array uniform. Max 64 obstacles.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct ObstacleArrayUniform {
    pub count: u32,
    pub _pad: [u32; 3],
    pub obstacles: [ObstacleGpu; 64],
}

impl ObstacleArrayUniform {
    pub fn from_obstacles(obstacles: &[Obstacle<Dim2>]) -> Self {
        let mut result = Self::zeroed();
        let count = obstacles.len().min(64);
        result.count = count as u32;
        for (i, obs) in obstacles.iter().take(64).enumerate() {
            result.obstacles[i] = ObstacleGpu::from_obstacle(obs);
        }
        result
    }
}
