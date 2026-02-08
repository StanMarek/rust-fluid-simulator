# AGENTS.md

Guidelines for AI coding agents working in this repository.

## Build & Run Commands

```bash
# Check entire workspace (fast, no codegen)
cargo check

# Check a single crate
cargo check -p sim-core
cargo check -p renderer
cargo check -p ui

# Lint (treat warnings seriously — fix all clippy warnings before committing)
cargo clippy --workspace

# Run desktop app (always --release for real perf; debug is 10-50x slower for SPH)
cargo run -p desktop --release

# Run benchmarks
cargo run -p benchmark --release -- --particles 50000 --steps 1000

# Build WASM (web)
cd apps/web && wasm-pack build --target web --out-dir pkg
```

There are **no tests yet**. When tests are added, they will use standard `cargo test`:
```bash
cargo test --workspace           # all tests
cargo test -p sim-core           # single crate
cargo test -p sim-core -- test_name  # single test by name
```

## Architecture Overview

Cargo workspace with `resolver = "2"`. Crate dependency graph:

```
apps/desktop, apps/web → ui → renderer → common
                           → sim-core → common
                           → sim-gpu → sim-core, common
tools/benchmark → sim-core, common
```

All simulation code is generic over `D: Dimension` (Dim2/Dim3). Only Dim2 is active.
Particle data uses SoA layout (`ParticleStorage<D>`). `nalgebra` for sim math, `glam` for rendering, `bytemuck` for GPU buffers.

## Code Style

### Imports

Three groups separated by blank lines, each group sorted alphabetically:
1. Standard library (`std::*`)
2. External/workspace crates (`common::`, `nalgebra::`, `wgpu::`, etc.)
3. Crate-internal (`crate::`, `super::`)

```rust
use std::collections::HashMap;

use common::{Dimension, SimConfig};
use wgpu::util::DeviceExt;

use crate::particle::ParticleStorage;
use crate::solver;
```

- Brace-group multiple items: `use common::{Dim2, Dimension, SimConfig};`
- No wildcard imports (`use foo::*`)
- No `pub(crate)` or `pub(super)` — use `pub` or private only

### Naming

| Kind | Convention | Examples |
|------|-----------|----------|
| Types/structs | `PascalCase` | `SimConfig`, `ParticleStorage`, `Camera2D` |
| Enums | `PascalCase` | `SimEvent`, `Tool`, `ColorMapType` |
| Functions/methods | `snake_case` | `load_scene()`, `spawn_particles_at()` |
| Constants | `SCREAMING_SNAKE_CASE` | `QUAD_VERTICES`, `DEFAULT_PARTICLE_RADIUS` |
| Modules | `snake_case` | `particle_renderer`, `color_map` |
| Unused params | Prefix `_` | `_config`, `_radius`, `_frame` |
| UI draw fns | Prefix `draw_` | `draw_toolbar()`, `draw_properties()` |

### Formatting

- `f32` everywhere (no `f64`)
- Float literals: `0.0`, `1.0`, `-9.81`, `1e-10`; use suffix `1.0_f32` only when inference needs help
- `usize` for indices/counts, `u64` for step counts, `u32` for wgpu-facing values
- `[f32; 2]` / `[f32; 3]` for serialization boundaries (config, emitters, GPU uniforms)
- Use `Self` in constructors instead of repeating the type name
- Prefer `self.field` shorthand in struct literals when variable name matches

### Generics

Bounds go directly on the type/function (not `where` clauses):
```rust
pub struct Simulation<D: Dimension> { ... }
pub fn enforce_boundaries<D: Dimension>(particles: &mut ParticleStorage<D>, config: &SimConfig) { ... }
```

Use turbofish for generic function calls:
```rust
solver::apply_gravity::<D>(&mut self.particles, &self.config);
Simulation::<Dim2>::new(config);
```

Use `PhantomData<D>` when a generic parameter is only used through associated items.

### Derives

Order: `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, `Serialize`, `Deserialize`, bytemuck.

| Pattern | When |
|---------|------|
| `#[derive(Debug, Clone)]` | Mutable domain structs |
| `#[derive(Debug, Clone, Copy, PartialEq, Eq)]` | Small enums, value types |
| `#[derive(Debug, Clone, Serialize, Deserialize)]` | Serializable config/scene types |
| `#[repr(C)]` + `#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]` | GPU buffer structs |

### Error Handling

- Simulation/physics code is **infallible** — no `Result` returns, just math on in-memory data
- Use `Option` for fallible lookups; chain with `.map_or()`, `.and_then()`, `.ok()?`
- Defensive defaults over panics: `config.particle_mass > 0.0` checks, `.clamp()`, `.max(1e-6)`
- No `unwrap()` or `expect()` in library code
- Custom error enums with manual `From` impls (no `thiserror`) for I/O boundaries

### Documentation

- Every public item gets a `///` doc comment — one or two sentences, capital letter, period
- Document physics meaning and units on fields: `/// Smoothing radius (h). Determines kernel support.`
- Numbered inline comments for algorithm steps: `// 1. Clear accelerations`
- Phase/roadmap comments for stubs: `// Future: Surface tension computation.`
- No `//!` module-level docs — use `///` on the module's primary type instead

### Module Organization

- Nested modules use `mod.rs` files (not `module_name.rs` alongside `module_name/`)
- `lib.rs` pattern: `pub mod` declarations (alphabetical) → blank line → `pub use` re-exports
- One primary type per file, with its `impl` blocks
- `impl` block order: struct definition → inherent impl → trait impls (`Default`, `App`, etc.)
- Cargo.toml deps: internal workspace paths first, then external workspace deps

### Logging

- Use `log` crate (not `tracing`), initialized with `env_logger` at binary entry points
- Keep logging sparse — only significant lifecycle events (`log::info!`)
- No `debug!`, `warn!`, or `error!` in current codebase

### Safety

- **No `unsafe` code.** Use `bytemuck::cast_slice()` for GPU buffer conversions
- `#[repr(C)]` + `bytemuck::Pod` + `bytemuck::Zeroable` for memory layout guarantees
- No `#![allow(...)]` at file level; use field-level `#[allow(dead_code)]` with a comment when needed

## Key Domain Rules

- SPH parameter consistency: `h ≈ 1.2 * particle_spacing`
- WCSPH: speed of sound ≥ 10x max fluid velocity
- Fixed timestep (`dt = 0.0002`) with accumulator pattern
- Always `--release` for performance testing
- Domain boundaries use clamp + velocity reflection with damping factor
