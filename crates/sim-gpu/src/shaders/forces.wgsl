// Pressure + viscosity force compute shader — Phase 4
// Placeholder: computes pressure gradient and viscosity forces.

@group(0) @binding(0)
var<storage, read> positions: array<vec2<f32>>;

@group(0) @binding(1)
var<storage, read> densities: array<f32>;

@group(0) @binding(2)
var<storage, read_write> accelerations: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // TODO: Implement pressure and viscosity forces
    accelerations[id.x] = vec2<f32>(0.0, -9.81);
}
