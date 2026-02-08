// Future: Field visualization shader (velocity arrows, pressure contours).
// Placeholder — will be implemented in Phase 6.

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
