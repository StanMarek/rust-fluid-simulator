// Combined pressure gradient + viscosity force compute shader.
// Avoids redundant neighbor traversal by computing both in one pass.

@group(0) @binding(0)
var<uniform> params: SimParams;

@group(0) @binding(1)
var<storage, read> positions: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read> velocities: array<vec2<f32>>;

@group(0) @binding(3)
var<storage, read> densities: array<f32>;

@group(0) @binding(4)
var<storage, read> pressures: array<f32>;

@group(0) @binding(5)
var<storage, read> masses: array<f32>;

@group(0) @binding(6)
var<storage, read_write> accelerations: array<vec2<f32>>;

// Grid buffers (bind group 1)
@group(1) @binding(0)
var<storage, read> sorted_indices: array<u32>;

@group(1) @binding(1)
var<storage, read> cell_start: array<u32>;

@group(1) @binding(2)
var<storage, read> cell_count: array<u32>;

// Cubic spline kernel gradient for 2D
fn kernel_grad_w(r_vec: vec2<f32>, r: f32) -> vec2<f32> {
    if r < 1e-10 {
        return vec2<f32>(0.0, 0.0);
    }

    let q = r * params.h_inv;
    let grad_q = r_vec / (r * params.h);

    var dw_dq: f32;
    if q <= 1.0 {
        dw_dq = -3.0 * q + 2.25 * q * q;
    } else if q <= 2.0 {
        let t = 2.0 - q;
        dw_dq = -0.75 * t * t;
    } else {
        return vec2<f32>(0.0, 0.0);
    }

    return grad_q * (params.sigma * dw_dq);
}

// Cubic spline kernel Laplacian for 2D
fn kernel_laplacian_w(r: f32) -> f32 {
    let q = r * params.h_inv;

    var d2w_dq2: f32;
    if q <= 1.0 {
        d2w_dq2 = -3.0 + 4.5 * q;
    } else if q <= 2.0 {
        d2w_dq2 = 1.5 * (2.0 - q);
    } else {
        return 0.0;
    }

    return params.sigma * d2w_dq2 / params.h_sq;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.particle_count {
        return;
    }

    let pos_i = positions[i];
    let vel_i = velocities[i];
    let rho_i = max(densities[i], 1e-6);
    let p_i = pressures[i];

    var f_pressure = vec2<f32>(0.0, 0.0);
    var f_viscosity = vec2<f32>(0.0, 0.0);

    let cell = grid_cell(pos_i, params.grid_cell_size);

    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            let h = hash_coords(cell.x + dx, cell.y + dy, params.grid_table_size);
            let start = cell_start[h];
            let count_val = cell_count[h];

            for (var k = 0u; k < count_val; k++) {
                let j = sorted_indices[start + k];
                if j == i {
                    continue;
                }

                let diff = pos_i - positions[j];
                let r_sq = dot(diff, diff);

                if r_sq < params.support_sq && r_sq > 1e-12 {
                    let r = sqrt(r_sq);
                    let m_j = masses[j];
                    let rho_j = max(densities[j], 1e-6);
                    let p_j = pressures[j];

                    // Pressure gradient: -m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_W
                    let pressure_term = p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j);
                    let grad_w = kernel_grad_w(diff, r);
                    f_pressure -= grad_w * (m_j * pressure_term);

                    // Viscosity: mu * m_j * (v_j - v_i) / rho_j * laplacian_W
                    let lap_w = kernel_laplacian_w(r);
                    let vel_diff = velocities[j] - vel_i;
                    f_viscosity += vel_diff * (params.viscosity * m_j * lap_w / rho_j);
                }
            }
        }
    }

    // Add forces to existing acceleration (gravity was already set)
    accelerations[i] += f_pressure + f_viscosity;
}
