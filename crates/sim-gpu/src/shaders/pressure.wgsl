// Pressure computation (Tait equation of state).
// Per-particle: p = k * ((rho/rho0)^7 - 1), clamped >= 0

@group(0) @binding(0)
var<uniform> params: SimParams;

@group(0) @binding(1)
var<storage, read> densities: array<f32>;

@group(0) @binding(2)
var<storage, read_write> pressures: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.particle_count {
        return;
    }

    let rho = densities[i];
    let rho_ratio = rho / params.rest_density;

    // (rho/rho0)^7 = r^4 * r^2 * r
    let r2 = rho_ratio * rho_ratio;
    let r4 = r2 * r2;
    let r7 = r4 * r2 * rho_ratio;

    let p = params.stiffness * (r7 - 1.0);
    pressures[i] = max(p, 0.0);
}
