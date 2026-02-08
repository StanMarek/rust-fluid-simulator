// Shared simulation parameters struct — prepended to all compute shaders.

struct SimParams {
    // Row 0: kernel params
    h: f32,
    h_sq: f32,
    h_inv: f32,
    sigma: f32,

    // Row 1: SPH params
    support_sq: f32,
    rest_density: f32,
    stiffness: f32,
    viscosity: f32,

    // Row 2: integration params
    gravity: vec2<f32>,
    dt: f32,
    boundary_damping: f32,

    // Row 3: domain bounds
    domain_min: vec2<f32>,
    domain_max: vec2<f32>,

    // Row 4: grid + particle info
    particle_count: u32,
    grid_table_size: u32,
    grid_cell_size: f32,
    _pad: f32,
}

struct ObstacleData {
    obstacle_type: u32,  // 0 = Circle, 1 = Box
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    data: vec4<f32>,     // Circle: (cx, cy, r, 0), Box: (min_x, min_y, max_x, max_y)
}

struct ObstacleArray {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    obstacles: array<ObstacleData, 64>,
}
