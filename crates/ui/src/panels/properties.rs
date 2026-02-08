use common::SimConfig;
use egui::Ui;

use crate::theme;

/// Draw a compact labeled slider: label on the left, slider on the right.
fn labeled_slider(ui: &mut Ui, label: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>, logarithmic: bool) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).size(11.0).color(theme::TEXT_SECONDARY));
        let mut slider = egui::Slider::new(value, range).show_value(true);
        if logarithmic {
            slider = slider.logarithmic(true);
        }
        ui.add(slider);
    });
}

/// Draw the properties panel with simulation parameter sliders in collapsible sections.
pub fn draw_properties(ui: &mut Ui, config: &mut SimConfig) {
    // Physics section
    let physics_id = ui.make_persistent_id("properties_physics");
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), physics_id, true)
        .show_header(ui, |ui| {
            ui.label(theme::section_heading("Physics"));
        })
        .body(|ui| {
            theme::section_frame().show(ui, |ui| {
                labeled_slider(ui, "Smoothing Radius", &mut config.smoothing_radius, 0.01..=0.5, true);
                labeled_slider(ui, "Stiffness", &mut config.stiffness, 10.0..=100000.0, true);
                labeled_slider(ui, "Viscosity", &mut config.viscosity, 0.001..=10.0, true);
                labeled_slider(ui, "Rest Density", &mut config.rest_density, 100.0..=5000.0, false);
            });
        });

    ui.add_space(theme::SECTION_SPACING);

    // Gravity section
    let gravity_id = ui.make_persistent_id("properties_gravity");
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), gravity_id, true)
        .show_header(ui, |ui| {
            ui.label(theme::section_heading("Gravity"));
        })
        .body(|ui| {
            theme::section_frame().show(ui, |ui| {
                labeled_slider(ui, "X", &mut config.gravity[0], -20.0..=20.0, false);
                labeled_slider(ui, "Y", &mut config.gravity[1], -20.0..=20.0, false);
            });
        });

    ui.add_space(theme::SECTION_SPACING);

    // Time Step section
    let dt_id = ui.make_persistent_id("properties_dt");
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), dt_id, false)
        .show_header(ui, |ui| {
            ui.label(theme::section_heading("Time Step"));
        })
        .body(|ui| {
            theme::section_frame().show(ui, |ui| {
                labeled_slider(ui, "dt", &mut config.time_step, 0.0001..=0.01, true);
            });
        });

    ui.add_space(theme::SECTION_SPACING);

    // Boundary section
    let boundary_id = ui.make_persistent_id("properties_boundary");
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), boundary_id, false)
        .show_header(ui, |ui| {
            ui.label(theme::section_heading("Boundary"));
        })
        .body(|ui| {
            theme::section_frame().show(ui, |ui| {
                labeled_slider(ui, "Damping", &mut config.boundary_damping, 0.0..=1.0, false);
            });
        });
}
