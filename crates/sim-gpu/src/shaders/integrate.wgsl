// Integration + boundary + obstacle enforcement compute shader.
// Replaces CPU steps 8 (Euler) + 9 (boundaries) + 10 (obstacles).

@group(0) @binding(0)
var<uniform> params: SimParams;

@group(0) @binding(1)
var<storage, read_write> positions: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read_write> velocities: array<vec2<f32>>;

@group(0) @binding(3)
var<storage, read> accelerations: array<vec2<f32>>;

@group(0) @binding(4)
var<uniform> obstacles: ObstacleArray;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.particle_count {
        return;
    }

    var vel = velocities[i];
    var pos = positions[i];
    let acc = accelerations[i];

    // Symplectic Euler: v += a*dt, x += v*dt
    vel += acc * params.dt;
    pos += vel * params.dt;

    // Domain boundary enforcement (clamp + velocity reflection)
    let dmin = params.domain_min;
    let dmax = params.domain_max;
    let damp = params.boundary_damping;

    // X axis
    if pos.x < dmin.x {
        pos.x = dmin.x;
        vel.x = -vel.x * damp;
    } else if pos.x > dmax.x {
        pos.x = dmax.x;
        vel.x = -vel.x * damp;
    }

    // Y axis
    if pos.y < dmin.y {
        pos.y = dmin.y;
        vel.y = -vel.y * damp;
    } else if pos.y > dmax.y {
        pos.y = dmax.y;
        vel.y = -vel.y * damp;
    }

    // Obstacle enforcement
    for (var o = 0u; o < obstacles.count; o++) {
        let obs = obstacles.obstacles[o];
        var dist: f32;
        var normal: vec2<f32>;

        if obs.obstacle_type == 0u {
            // Circle: data = (cx, cy, r, 0)
            let center = obs.data.xy;
            let radius = obs.data.z;
            let diff = pos - center;
            let d = length(diff);
            dist = d - radius;
            if d > 1e-10 {
                normal = diff / d;
            } else {
                normal = vec2<f32>(0.0, 1.0);
            }
        } else {
            // Box: data = (min_x, min_y, max_x, max_y)
            let bmin = obs.data.xy;
            let bmax = obs.data.zw;

            // SDF for axis-aligned box
            let d_lo_x = bmin.x - pos.x;
            let d_hi_x = pos.x - bmax.x;
            let d_lo_y = bmin.y - pos.y;
            let d_hi_y = pos.y - bmax.y;

            var max_neg = d_lo_x;
            normal = vec2<f32>(-1.0, 0.0);

            if d_hi_x > max_neg {
                max_neg = d_hi_x;
                normal = vec2<f32>(1.0, 0.0);
            }
            if d_lo_y > max_neg {
                max_neg = d_lo_y;
                normal = vec2<f32>(0.0, -1.0);
            }
            if d_hi_y > max_neg {
                max_neg = d_hi_y;
                normal = vec2<f32>(0.0, 1.0);
            }
            dist = max_neg;
        }

        if dist < 0.0 {
            // Push to surface
            pos += normal * (-dist + 1e-4);

            // Reflect velocity along normal
            let v_dot_n = dot(vel, normal);
            if v_dot_n < 0.0 {
                vel += normal * (-v_dot_n * (1.0 + damp));
            }
        }
    }

    positions[i] = pos;
    velocities[i] = vel;
}
