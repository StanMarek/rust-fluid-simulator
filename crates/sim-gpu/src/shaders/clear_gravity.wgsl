// Clear accelerations + apply gravity compute shader.
// Replaces CPU steps 1 (clear) + 7 (gravity): accelerations[i] = gravity

@group(0) @binding(0)
var<uniform> params: SimParams;

@group(0) @binding(1)
var<storage, read_write> accelerations: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.particle_count {
        return;
    }
    accelerations[i] = params.gravity;
}
