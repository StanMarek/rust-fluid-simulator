// Density computation compute shader — Phase 4
// Placeholder: computes SPH density for each particle.

@group(0) @binding(0)
var<storage, read> positions: array<vec2<f32>>;

@group(0) @binding(1)
var<storage, read_write> densities: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // TODO: Implement SPH density summation
    densities[id.x] = 1000.0;
}
