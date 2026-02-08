// Particle instanced rendering shader.
// Each particle is a quad with a circle drawn in the fragment shader.

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,    // Unit quad vertex [-1, 1]
    @location(1) world_pos: vec2<f32>,   // Particle center in world space
    @location(2) color: vec3<f32>,       // Particle color
    @location(3) radius: f32,            // Particle radius in world units
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,          // Local coordinates for circle SDF
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Scale quad by particle radius and translate to world position
    let world = vec4<f32>(
        input.world_pos + input.quad_pos * input.radius,
        0.0,
        1.0
    );

    out.clip_position = camera.view_proj * world;
    out.uv = input.quad_pos;
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Circle SDF: discard pixels outside the circle
    let dist = length(input.uv);
    if dist > 1.0 {
        discard;
    }

    // Smooth edge for anti-aliasing
    let alpha = 1.0 - smoothstep(0.85, 1.0, dist);

    return vec4<f32>(input.color, alpha);
}
