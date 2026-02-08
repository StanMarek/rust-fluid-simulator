use common::SimConfig;
use egui::Ui;

/// Draw the properties panel with simulation parameter sliders.
pub fn draw_properties(ui: &mut Ui, config: &mut SimConfig, particle_count: usize, fps: f32) {
    ui.heading("Properties");
    ui.separator();

    // Display info
    ui.label(format!("Particles: {}", particle_count));
    ui.label(format!("FPS: {:.1}", fps));
    ui.separator();

    // Physics parameters
    ui.label("Physics");
    ui.add(
        egui::Slider::new(&mut config.smoothing_radius, 0.01..=0.5)
            .text("Smoothing Radius")
            .logarithmic(true),
    );
    ui.add(
        egui::Slider::new(&mut config.stiffness, 10.0..=100000.0)
            .text("Stiffness")
            .logarithmic(true),
    );
    ui.add(
        egui::Slider::new(&mut config.viscosity, 0.001..=10.0)
            .text("Viscosity")
            .logarithmic(true),
    );
    ui.add(egui::Slider::new(&mut config.rest_density, 100.0..=5000.0).text("Rest Density"));

    ui.separator();
    ui.label("Gravity");
    ui.add(egui::Slider::new(&mut config.gravity[0], -20.0..=20.0).text("X"));
    ui.add(egui::Slider::new(&mut config.gravity[1], -20.0..=20.0).text("Y"));

    ui.separator();
    ui.label("Time Step");
    ui.add(
        egui::Slider::new(&mut config.time_step, 0.0001..=0.01)
            .text("dt")
            .logarithmic(true),
    );

    ui.separator();
    ui.label("Boundary");
    ui.add(egui::Slider::new(&mut config.boundary_damping, 0.0..=1.0).text("Damping"));
}
