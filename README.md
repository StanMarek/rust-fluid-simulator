# Fluid Simulator

Real-time 2D SPH (Smoothed Particle Hydrodynamics) fluid simulation with GPU acceleration.

## Quick Start

```bash
# Desktop (native)
cargo run -p desktop --release

# Benchmarks
cargo run -p benchmark --release -- --particles 50000 --steps 1000
```

## Architecture

Cargo workspace with the following crates:

- **common** — Shared types, dimension traits, configuration
- **sim-core** — CPU-based SPH simulation
- **sim-gpu** — GPU compute acceleration via wgpu
- **renderer** — wgpu-based particle visualization
- **ui** — egui interface panels
- **desktop** — Native desktop entry point
- **web** — WASM/WebGPU entry point
- **benchmark** — Headless performance benchmarks
