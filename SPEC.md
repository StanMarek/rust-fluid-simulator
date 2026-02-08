# Fluid Simulation App — Project Specification

## Overview

Build a real-time 2D fluid simulation application (with a clear path to 3D) targeting both desktop (native binary) and web (WASM) from a single Rust codebase. The tool is intended for both artists/designers and engineers.

---

## Key Decisions

- **Simulation method:** SPH (Smoothed Particle Hydrodynamics)
- **Dimensionality:** 2D now, architected for 3D extension via dimensional generics
- **Real-time:** Yes — target 60fps with ~50k–100k particles on GPU
- **Platforms:** Native desktop (Windows/macOS/Linux) + Web browser (WASM + WebGPU)
- **Target audience:** Both artists/designers and engineers

---

## Tech Stack

| Layer            | Choice                              |
| ---------------- | ----------------------------------- |
| Language         | Rust                                |
| GPU Compute      | wgpu compute shaders (WGSL)         |
| Rendering        | wgpu                                |
| UI               | egui / eframe                       |
| Web              | WASM (wasm-bindgen, wasm-pack)      |
| Desktop          | Native binary via eframe            |
| Neighbor search  | Uniform spatial hash grid (custom)  |
| Math             | nalgebra (sim) + glam (rendering)   |
| Serialization    | serde + serde_json                  |

---

## Project Structure

This is a Cargo workspace with multiple crates. Create every file and directory listed below.

```
fluid-sim/
├── Cargo.toml                  # Workspace root
├── README.md
│
├── crates/
│   ├── sim-core/               # Pure simulation logic (no rendering, no UI)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── math/
│   │       │   ├── mod.rs
│   │       │   ├── vector.rs       # 2D/3D vector abstractions
│   │       │   └── bounds.rs       # AABB, domain boundaries
│   │       ├── particle/
│   │       │   ├── mod.rs
│   │       │   ├── properties.rs   # Position, velocity, density, pressure
│   │       │   └── storage.rs      # SoA layout for cache/GPU friendliness
│   │       ├── sph/
│   │       │   ├── mod.rs
│   │       │   ├── kernel.rs       # Smoothing kernels (cubic spline, Wendland)
│   │       │   ├── density.rs      # Density computation
│   │       │   ├── pressure.rs     # Pressure solver (WCSPH or PCISPH)
│   │       │   ├── viscosity.rs    # Viscosity forces
│   │       │   └── surface.rs      # Surface tension (future, leave as empty stub)
│   │       ├── neighbor/
│   │       │   ├── mod.rs
│   │       │   ├── grid.rs         # Uniform spatial hash grid
│   │       │   └── query.rs        # Neighbor query interface
│   │       ├── solver.rs           # Time integration (symplectic Euler, leapfrog)
│   │       ├── boundary.rs         # Boundary handling (ghost particles, penalty)
│   │       ├── scene.rs            # Scene description (emitters, obstacles, domain)
│   │       └── simulation.rs       # Top-level sim state, step(), reset()
│   │
│   ├── sim-gpu/                # GPU compute acceleration
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── context.rs          # wgpu device/queue setup
│   │       ├── buffers.rs          # GPU buffer management for particle data
│   │       ├── pipelines/
│   │       │   ├── mod.rs
│   │       │   ├── neighbor.rs     # GPU neighbor search (hash + sort)
│   │       │   ├── density.rs      # Density compute pass
│   │       │   ├── forces.rs       # Pressure + viscosity compute pass
│   │       │   └── integrate.rs    # Position/velocity update pass
│   │       └── shaders/
│   │           ├── neighbor.wgsl
│   │           ├── density.wgsl
│   │           ├── forces.wgsl
│   │           └── integrate.wgsl
│   │
│   ├── renderer/               # Visualization layer
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── camera.rs           # 2D camera (pan, zoom) — extend to 3D orbit later
│   │       ├── particle_renderer.rs # Instanced circle/point rendering
│   │       ├── field_renderer.rs   # Velocity/pressure field overlays (future stub)
│   │       ├── grid_overlay.rs     # Debug: show spatial grid, domain bounds
│   │       ├── color_map.rs        # Map scalar fields to colors (velocity, pressure, density)
│   │       └── shaders/
│   │           ├── particle.wgsl
│   │           └── field.wgsl
│   │
│   ├── ui/                     # egui-based interface panels
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── app.rs              # Main app state, orchestrates sim + render + ui
│   │       ├── panels/
│   │       │   ├── mod.rs
│   │       │   ├── toolbar.rs      # Tool selection (emit, drag, erase, etc.)
│   │       │   ├── properties.rs   # SPH parameters (viscosity, stiffness, etc.)
│   │       │   ├── scene.rs        # Scene setup (emitters, boundaries, obstacles)
│   │       │   ├── viewport.rs     # Sim viewport (egui <-> wgpu integration)
│   │       │   └── timeline.rs     # Play/pause/step, speed control
│   │       ├── interaction.rs      # Mouse to world coords, particle spawning
│   │       └── theme.rs            # UI styling
│   │
│   └── common/                 # Shared types and traits
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── dimension.rs        # Dim2 / Dim3 trait for 2D->3D generics
│           ├── config.rs           # Simulation config (serializable with serde)
│           ├── events.rs           # Sim <-> UI communication
│           └── export.rs           # Particle data export (CSV, VTK — future stub)
│
├── apps/
│   ├── desktop/                # Native entry point
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── main.rs            # eframe::run_native(...)
│   │
│   └── web/                    # WASM entry point
│       ├── Cargo.toml
│       ├── src/
│       │   └── main.rs            # eframe WASM bootstrap
│       ├── index.html
│       └── build.sh               # wasm-pack build script
│
├── assets/                     # Presets, icons, example scenes
│   ├── scenes/
│   │   ├── dam_break.json
│   │   └── double_emitter.json
│   └── icons/
│
└── tools/
    └── benchmark/              # Headless performance benchmarks
        ├── Cargo.toml
        └── src/
            └── main.rs
```

---

## Workspace Cargo.toml

```toml
[workspace]
resolver = "2"
members = [
    "crates/common",
    "crates/sim-core",
    "crates/sim-gpu",
    "crates/renderer",
    "crates/ui",
    "apps/desktop",
    "apps/web",
    "tools/benchmark",
]

[workspace.dependencies]
nalgebra = "0.33"
glam = "0.29"
wgpu = "24"
egui = "0.31"
eframe = "0.31"
bytemuck = { version = "1", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
log = "0.4"
env_logger = "0.11"
wasm-bindgen = "0.2"
```

---

## Architecture Principles

### 1. Dimensional Generics (2D → 3D)

Define a `Dimension` trait in `common/dimension.rs` that abstracts over 2D and 3D:

```rust
pub trait Dimension: 'static + Send + Sync + Clone {
    type Vector: Copy + Default + Send + Sync; // nalgebra Vector2<f32> or Vector3<f32>
    const DIM: usize;
}

#[derive(Clone)]
pub struct Dim2;

#[derive(Clone)]
pub struct Dim3;

impl Dimension for Dim2 {
    type Vector = nalgebra::Vector2<f32>;
    const DIM: usize = 2;
}

impl Dimension for Dim3 {
    type Vector = nalgebra::Vector3<f32>;
    const DIM: usize = 3;
}
```

All simulation code in `sim-core` should be generic over `D: Dimension` so that adding 3D support later does not require a rewrite.

### 2. Particle Storage (Struct of Arrays)

Use SoA (Struct of Arrays) layout in `particle/storage.rs` for cache-friendly access and easy GPU buffer mapping:

```rust
pub struct ParticleStorage<D: Dimension> {
    pub positions: Vec<D::Vector>,
    pub velocities: Vec<D::Vector>,
    pub accelerations: Vec<D::Vector>,
    pub densities: Vec<f32>,
    pub pressures: Vec<f32>,
    pub masses: Vec<f32>,
}
```

### 3. SimBackend Trait

Define a trait in `common` that both the CPU solver (`sim-core`) and GPU solver (`sim-gpu`) implement:

```rust
pub trait SimBackend<D: Dimension> {
    fn step(&mut self, particles: &mut ParticleStorage<D>, config: &SimConfig, dt: f32);
    fn reset(&mut self);
}
```

This allows the app to switch between CPU and GPU backends at runtime.

### 4. Separation of Concerns

- `sim-core`: Pure physics. No rendering, no UI, no platform dependencies. Must compile to WASM.
- `sim-gpu`: GPU acceleration via wgpu compute shaders. Mirrors `sim-core` logic.
- `renderer`: Takes particle data, draws it. Knows nothing about UI panels.
- `ui`: Composes renderer output into egui panels. Orchestrates everything in `app.rs`.
- `apps/desktop` and `apps/web`: Thin entry-point shells only. All real logic lives in `ui::app`.

---

## SPH Algorithm Implementation Details

### Smoothing Kernels (`sph/kernel.rs`)

Implement at minimum the **cubic spline kernel** (M4) as the default. Provide a `Kernel` trait so other kernels (Wendland C2, poly6, spiky) can be swapped in:

```rust
pub trait SmoothingKernel<D: Dimension> {
    fn w(&self, r: f32, h: f32) -> f32;           // Kernel value
    fn grad_w(&self, r_vec: D::Vector, r: f32, h: f32) -> D::Vector; // Kernel gradient
    fn laplacian_w(&self, r: f32, h: f32) -> f32;  // Kernel laplacian (for viscosity)
}
```

### Simulation Loop (per timestep)

1. **Neighbor search** — Build/update spatial hash grid, query neighbors within smoothing radius `h`
2. **Density computation** — For each particle, sum kernel contributions from neighbors
3. **Pressure computation** — Equation of state: `p = k * ((ρ/ρ₀)^γ - 1)` (Tait equation for WCSPH)
4. **Force computation** — Pressure gradient force + viscosity force + gravity
5. **Time integration** — Symplectic Euler or leapfrog to update velocity and position
6. **Boundary enforcement** — Clamp positions, reflect velocities, or apply penalty forces

### Neighbor Search (`neighbor/grid.rs`)

Uniform spatial hash grid with cell size = smoothing radius `h`. For 2D:
- Hash: `hash(ix, iy) = (ix * 73856093) ^ (iy * 19349663) % table_size`
- Store particle indices in each cell
- Query: check the current cell and all adjacent cells (9 cells in 2D, 27 in 3D)

### Default Simulation Parameters (`common/config.rs`)

```rust
pub struct SimConfig {
    pub smoothing_radius: f32,     // h = 0.1
    pub rest_density: f32,         // ρ₀ = 1000.0 (water)
    pub stiffness: f32,            // k = 1000.0
    pub viscosity: f32,            // μ = 0.1
    pub gravity: [f32; 2],         // [0.0, -9.81]
    pub time_step: f32,            // dt = 0.001 (fixed)
    pub particle_mass: f32,        // auto-computed from spacing and rest density
    pub domain_min: [f32; 2],      // [0.0, 0.0]
    pub domain_max: [f32; 2],      // [1.0, 1.0]
    pub boundary_damping: f32,     // 0.5
}
```

---

## Rendering Details

### Particle Rendering (`renderer/particle_renderer.rs`)

Use instanced rendering. Each particle is a small quad with a circle shader (or just GL_POINTS with point size for simplicity). Pass particle positions and a scalar field (velocity magnitude, pressure, density) to the GPU. The fragment shader uses `color_map.rs` logic to map scalars to colors.

### Color Maps (`renderer/color_map.rs`)

Provide at least: viridis, plasma, coolwarm, and a simple blue-white (water). User should be able to select which scalar field to visualize and which color map to use.

### Camera (`renderer/camera.rs`)

2D camera with pan (middle mouse drag) and zoom (scroll wheel). Store as view matrix. Later extend to 3D orbit camera.

---

## UI Panels

### Toolbar (`ui/panels/toolbar.rs`)
- **Emit tool**: Click/drag to spawn particles
- **Drag tool**: Click/drag to apply force to nearby particles
- **Erase tool**: Click to remove particles in radius
- **Obstacle tool**: Place static boundary objects (future)

### Properties Panel (`ui/panels/properties.rs`)
- Sliders for: smoothing radius, stiffness, viscosity, gravity, time step
- Dropdown for: pressure solver (WCSPH / PCISPH future)
- Particle count display
- FPS / simulation time display

### Timeline (`ui/panels/timeline.rs`)
- Play / Pause / Step buttons
- Simulation speed multiplier slider
- Reset button

### Viewport (`ui/panels/viewport.rs`)
- The main simulation view
- Integrates wgpu rendering surface with egui
- Handles mouse input for tools

---

## Build & Run

### Desktop

```bash
cargo run -p desktop --release
```

### Web

```bash
cd apps/web
wasm-pack build --target web
# Serve index.html with a local server
python3 -m http.server 8080
```

### Benchmarks

```bash
cargo run -p benchmark --release -- --particles 50000 --steps 1000
```

---

## Implementation Order

Build in this exact order. Each step should compile and run before moving to the next.

### Phase 1: Foundation (get particles on screen)

1. **`crates/common`** — Dimension trait (`Dim2`, `Dim3`), `SimConfig` with serde, `SimBackend` trait, event types
2. **`crates/sim-core`** — Particle storage (SoA), gravity-only solver (no SPH yet), simple boundary reflection. Particles should fall and bounce.
3. **`crates/renderer`** — wgpu setup, instanced circle rendering, 2D camera with pan/zoom, basic color mapping
4. **`crates/ui`** — Minimal egui app with a viewport panel showing the renderer, play/pause button, particle count label
5. **`apps/desktop`** — Wire everything together with `eframe::run_native`. **Milestone: window opens, particles fall and bounce.**

### Phase 2: SPH Physics (make it a fluid)

6. **Neighbor search** — Implement uniform hash grid in `sim-core/neighbor/`
7. **Smoothing kernels** — Cubic spline kernel with gradient and laplacian
8. **Density computation** — Sum kernel contributions from neighbors
9. **Pressure forces** — Tait equation + pressure gradient force
10. **Viscosity forces** — Artificial viscosity
11. **Full SPH step** — Integrate all forces. **Milestone: dam break simulation works on CPU.**

### Phase 3: Interactivity (make it a tool)

12. **Emit tool** — Click/drag to spawn particles
13. **Drag tool** — Apply forces with mouse
14. **Properties panel** — Sliders for all sim parameters
15. **Color mapping** — Visualize velocity/pressure/density fields
16. **Scene presets** — Load dam_break.json, double_emitter.json

### Phase 4: GPU Acceleration

17. **`crates/sim-gpu`** — Port neighbor search to compute shader
18. **GPU density + forces** — Port SPH compute passes to WGSL
19. **GPU integration** — Full simulation loop on GPU
20. **Backend switching** — UI toggle between CPU and GPU backends

### Phase 5: Web Deployment

21. **`apps/web`** — WASM entry point, index.html, build script
22. **Test WebGPU** — Verify compute shaders and rendering work in browser
23. **Optimize WASM** — Minimize bundle size, handle WebGPU fallback

### Phase 6: Polish & Extend

24. **Surface tension** — Implement in `sph/surface.rs`
25. **Obstacle tool** — Static boundary objects (circles, boxes, arbitrary shapes)
26. **Export** — Particle data to CSV/VTK
27. **3D mode** — Instantiate `Dim3`, add 3D camera, 3D rendering (marching cubes or screen-space fluid)
28. **Debug overlays** — Spatial grid visualization, velocity arrows, pressure contours

---

## Notes

- Always use `--release` for performance testing. Debug builds will be 10-50x slower for SPH.
- Fixed timestep (`dt = 0.001`) is recommended for SPH stability. Use an accumulator pattern in the main loop to decouple simulation and rendering rates.
- SPH is sensitive to parameter tuning. Provide reasonable defaults and expose everything via UI sliders for experimentation.
- The smoothing radius `h` and particle spacing should be consistent: typically `h = 1.2 * particle_spacing`.
- For WCSPH, the speed of sound `c` should be at least 10x the maximum expected fluid velocity to maintain low compressibility.
