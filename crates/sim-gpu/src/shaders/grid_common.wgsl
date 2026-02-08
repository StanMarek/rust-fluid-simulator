// Shared grid functions — prepended to shaders that need neighbor search.

const HASH_PRIME_X: u32 = 73856093u;
const HASH_PRIME_Y: u32 = 19349663u;

fn grid_cell(pos: vec2<f32>, cell_size: f32) -> vec2<i32> {
    return vec2<i32>(floor(pos / cell_size));
}

fn hash_coords(cx: i32, cy: i32, table_size: u32) -> u32 {
    let hx = u32(cx) * HASH_PRIME_X;
    let hy = u32(cy) * HASH_PRIME_Y;
    return (hx ^ hy) % table_size;
}
