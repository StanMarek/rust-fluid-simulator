# Performance Optimization Design

**Date:** 2026-02-08
**Goal:** Achieve stable 30 FPS at ~1,200 particles (currently ~3 FPS)

## Problem Analysis

The simulator runs at ~3 FPS with 1,208 particles. Profiling analysis reveals three
categories of bottleneck:

### 1. Allocation Pressure (~60% of frame time, estimated)

The `SpatialHashGrid` neighbor search produces ~30n heap allocations per simulation step:
- `query_neighbors()` allocates a `Vec<usize>` per call (3n per step)
- `cell_coords()` allocates a `Vec<i64>` per call (3n per step)
- `visit_adjacent_cells()` allocates 9 `vec![]` per call (27n per step)
- `HashMap<u64, Vec<usize>>` is rebuilt from scratch every step

With 4 substeps/frame, that is ~145,000 heap allocations per frame for 1,200 particles.

### 2. Redundant Computation (~20% of frame time, estimated)

- Kernel normalization `sigma` recomputed on every `W()`, `grad_W()`, `laplacian_W()` call
- `powf(7.0)` in Tait EOS instead of integer multiplication (6 multiplies)
- No early distance rejection: kernel evaluates for particles up to ~4.2h away when only
  those within 2h contribute. ~89% of candidate pairs have W=0
- Temporary `Vec<D::Vector>` force buffers allocated per step in pressure/viscosity passes

### 3. Rendering (~20% of frame time, estimated)

Each particle drawn individually via `egui::Painter::circle_filled()`. The wgpu instanced
renderer (`ParticleRenderer`) exists but is not wired into the application.

## Design

### Phase 1: Allocation-Free Neighbor Search

Replace `HashMap<u64, Vec<usize>>` with a flat sorted-array spatial hash.

**Data structure:**
```rust
pub struct SpatialHashGrid<D: Dimension> {
    /// Sorted (hash, particle_index) pairs.
    entries: Vec<(u32, u32)>,
    /// Offset table: offsets[hash] = (start, count) into entries.
    offsets: Vec<(u32, u32)>,
    cell_size: f32,
    table_size: u32,
    _marker: PhantomData<D>,
}
```

**Build phase (O(n)):**
1. For each particle, compute hash -> store `(hash, index)` in `entries`
2. Sort `entries` by hash (counting sort since hashes are bounded integers)
3. Build `offsets` table: scan sorted `entries`, record start/count per hash

**Query phase (zero allocations):**
```rust
fn query_neighbors(&self, pos: &D::Vector) -> impl Iterator<Item = usize> {
    // Visit 9 adjacent cells (2D), for each:
    //   hash = compute_hash(cell_coords)
    //   (start, count) = self.offsets[hash]
    //   iterate entries[start..start+count]
    // All on the stack, return iterator
}
```

**Cell coord helpers return fixed arrays:**
- `cell_coords(pos) -> [i64; 2]` (stack, not `Vec<i64>`)
- `adjacent_cells(center) -> [[i64; 2]; 9]` (stack, not `Vec<Vec<i64>>`)

**Files modified:**
- `crates/sim-core/src/neighbor/grid.rs` -- complete rewrite
- `crates/sim-core/src/sph/density.rs` -- use new query API
- `crates/sim-core/src/sph/pressure.rs` -- use new query API
- `crates/sim-core/src/sph/viscosity.rs` -- use new query API

### Phase 2: SPH Computation Optimizations

**2a. Precompute kernel constant.**
Add a `KernelParams` struct or pass `sigma` alongside `h`:
```rust
pub struct KernelParams {
    pub h: f32,
    pub sigma: f32,       // 10.0 / (7.0 * PI * h * h)
    pub h_sq: f32,        // h * h
    pub support_sq: f32,  // (2.0 * h) * (2.0 * h)
}
```
Computed once at the start of `step()`, passed to all kernel functions.

**2b. Integer power in Tait EOS.**
Replace `rho_ratio.powf(gamma)` with:
```rust
let r2 = rho_ratio * rho_ratio;
let r4 = r2 * r2;
let r7 = r4 * r2 * rho_ratio;
```

**2c. Early distance rejection.**
Before evaluating the kernel, check `dist_sq >= support_sq` and skip.
Avoids computing `sqrt`, division, and kernel polynomial for ~89% of candidate pairs.

**2d. Eliminate temporary force buffers.**
In pressure and viscosity passes, accumulate directly into `accelerations[i]`
using indexed access instead of allocating a `Vec<D::Vector>`.

**Files modified:**
- `crates/sim-core/src/sph/kernel.rs` -- add `KernelParams`, refactor signatures
- `crates/sim-core/src/sph/density.rs` -- use `KernelParams`, add distance check
- `crates/sim-core/src/sph/pressure.rs` -- integer power, distance check, remove buffer
- `crates/sim-core/src/sph/viscosity.rs` -- distance check, remove buffer
- `crates/sim-core/src/simulation.rs` -- construct `KernelParams` at step start

### Phase 3: Rayon Parallelization

Add `rayon` dependency to `sim-core`. Parallelize the three O(n*k) passes.

**Density pass:** `par_iter` over particles, each reads positions/masses (shared),
writes to its own `densities[i]`.

**Pressure/viscosity forces:** `par_iter` with indexed access. Each particle `i`
reads shared data and writes only to `accelerations[i]`.

The grid build phase stays sequential (or use rayon parallel sort for the counting sort).

**Files modified:**
- `crates/sim-core/Cargo.toml` -- add `rayon` dependency
- `crates/sim-core/src/sph/density.rs` -- `par_iter`
- `crates/sim-core/src/sph/pressure.rs` -- `par_iter`
- `crates/sim-core/src/sph/viscosity.rs` -- `par_iter`

### Phase 4: Wire Up wgpu Instanced Renderer

**Integration approach:**
1. Initialize `ParticleRenderer` during `CreationContext` in eframe's `new()` callback,
   when device/queue/surface format are available
2. Each frame: call `build_instances()` -> `update_instances()` -> render via
   `egui::PaintCallback` with a custom `CallbackTrait` implementation
3. Remove the `egui::Painter::circle_filled()` loop

**Files modified:**
- `crates/ui/src/app.rs` -- renderer init, paint callback, remove circle_filled loop
- `crates/renderer/src/particle_renderer.rs` -- minor API adjustments if needed

## Implementation Order

| Phase | What | Expected Speedup | Risk |
|-------|------|-----------------|------|
| 1 | Allocation-free neighbor search | 5-10x | Medium |
| 2 | SPH computation optimizations | 1.5-2x | Low |
| 3 | Rayon parallelization | 2-4x | Low |
| 4 | wgpu instanced renderer | Eliminates render bottleneck | Medium |

**Conservative estimate:** Phases 1-2 should reach 30 FPS for 1,200 particles.
Phases 3-4 provide headroom and enable scaling to higher particle counts.

## Verification

Run `cargo run -p benchmark --release -- --particles 1200 --steps 1000` before and
after each phase to measure sim-only improvement. Run the desktop app to verify
visual correctness and measure actual FPS.

## What Stays The Same

- `ParticleStorage<D>` SoA layout
- `Dimension` trait generic
- 9-phase simulation step structure
- All public APIs
- Scene system, UI panels, interaction tools
