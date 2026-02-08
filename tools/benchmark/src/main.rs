use std::time::Instant;

use common::{Dim2, SimConfig};
use sim_core::scene::SceneDescription;
use sim_core::Simulation;

fn main() {
    env_logger::init();

    // Parse simple CLI args
    let args: Vec<String> = std::env::args().collect();
    let steps = parse_arg(&args, "--steps").unwrap_or(1000);
    let _particles = parse_arg(&args, "--particles").unwrap_or(50000);

    println!("Fluid Simulator Benchmark");
    println!("========================");

    // Create simulation with dam break scene
    let mut sim = Simulation::<Dim2>::new(SimConfig::default());
    let scene = SceneDescription::dam_break();
    sim.load_scene(&scene);

    println!("Particles: {}", sim.particles.len());
    println!("Steps: {}", steps);
    println!();

    // Run benchmark
    let start = Instant::now();
    for _ in 0..steps {
        sim.step();
    }
    let elapsed = start.elapsed();

    let ms_per_step = elapsed.as_secs_f64() * 1000.0 / steps as f64;
    let steps_per_sec = steps as f64 / elapsed.as_secs_f64();

    println!("Results:");
    println!("  Total time: {:.2?}", elapsed);
    println!("  Per step:   {:.3} ms", ms_per_step);
    println!("  Steps/sec:  {:.1}", steps_per_sec);
    println!("  Sim time:   {:.4} s", sim.time);
}

fn parse_arg(args: &[String], name: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}
