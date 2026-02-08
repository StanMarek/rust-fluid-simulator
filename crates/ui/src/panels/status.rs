use egui::Ui;

use crate::theme;

/// Draw the bottom status bar.
pub fn draw_status_bar(
    ui: &mut Ui,
    particle_count: usize,
    fps: f32,
    sim_time: f32,
    step_count: u64,
    is_playing: bool,
    backend_name: &str,
) {
    ui.horizontal_centered(|ui| {
        ui.spacing_mut().item_spacing.x = 16.0;

        // Playing/paused indicator
        let (indicator_color, indicator_label) = if is_playing {
            (theme::TRANSPORT_PLAYING, "RUNNING")
        } else {
            (theme::TEXT_DISABLED, "PAUSED")
        };

        let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
        ui.painter()
            .circle_filled(rect.center(), 3.5, indicator_color);

        ui.label(
            egui::RichText::new(indicator_label)
                .size(10.0)
                .color(indicator_color),
        );

        // Separator
        ui.label(egui::RichText::new("\u{00B7}").size(10.0).color(theme::TEXT_DISABLED));

        // Backend indicator
        ui.label(theme::value_text(backend_name));

        ui.label(egui::RichText::new("\u{00B7}").size(10.0).color(theme::TEXT_DISABLED));

        // Particle count
        ui.label(egui::RichText::new("Particles").size(10.0).color(theme::TEXT_DISABLED));
        ui.label(theme::value_text(&format!("{}", particle_count)));

        ui.label(egui::RichText::new("\u{00B7}").size(10.0).color(theme::TEXT_DISABLED));

        // FPS
        ui.label(egui::RichText::new("FPS").size(10.0).color(theme::TEXT_DISABLED));
        ui.label(theme::value_text(&format!("{:.0}", fps)));

        ui.label(egui::RichText::new("\u{00B7}").size(10.0).color(theme::TEXT_DISABLED));

        // Sim time
        ui.label(egui::RichText::new("Time").size(10.0).color(theme::TEXT_DISABLED));
        ui.label(theme::value_text(&format!("{:.3}s", sim_time)));

        ui.label(egui::RichText::new("\u{00B7}").size(10.0).color(theme::TEXT_DISABLED));

        // Step count
        ui.label(egui::RichText::new("Steps").size(10.0).color(theme::TEXT_DISABLED));
        ui.label(theme::value_text(&format!("{}", step_count)));
    });
}
