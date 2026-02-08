// SPH density computation compute shader.
// Each thread computes one particle's density via kernel summation over neighbors.

@group(0) @binding(0)
var<uniform> params: SimParams;

@group(0) @binding(1)
var<storage, read> positions: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read> masses: array<f32>;

@group(0) @binding(3)
var<storage, read_write> densities: array<f32>;

// Grid buffers (bind group 1)
@group(1) @binding(0)
var<storage, read> sorted_indices: array<u32>;

@group(1) @binding(1)
var<storage, read> cell_start: array<u32>;

@group(1) @binding(2)
var<storage, read> cell_count: array<u32>;

// Cubic spline kernel W(r, h) for 2D
fn kernel_w(r: f32) -> f32 {
    let q = r * params.h_inv;
    if q <= 1.0 {
        return params.sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
    } else if q <= 2.0 {
        let t = 2.0 - q;
        return params.sigma * 0.25 * t * t * t;
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.particle_count {
        return;
    }

    let pos_i = positions[i];
    var density = 0.0;

    // Get grid cell for this particle
    let cell = grid_cell(pos_i, params.grid_cell_size);

    // Iterate 3x3 neighbor cells
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            let h = hash_coords(cell.x + dx, cell.y + dy, params.grid_table_size);
            let start = cell_start[h];
            let count_val = cell_count[h];

            for (var k = 0u; k < count_val; k++) {
                let j = sorted_indices[start + k];
                let diff = pos_i - positions[j];
                let r_sq = dot(diff, diff);

                if r_sq < params.support_sq {
                    let r = sqrt(r_sq);
                    density += masses[j] * kernel_w(r);
                }
            }
        }
    }

    densities[i] = density;
}
