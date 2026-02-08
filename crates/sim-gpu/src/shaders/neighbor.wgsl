// Neighbor search compute shader — Phase 4
// Placeholder: builds spatial hash grid on the GPU.

@group(0) @binding(0)
var<storage, read> positions: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // TODO: Implement spatial hash construction
    _ = positions[id.x];
}
