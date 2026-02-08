use egui::{CornerRadius, Sense, Stroke, Ui, Vec2};

use crate::theme;

/// Timeline state.
pub struct TimelineState {
    pub is_playing: bool,
    pub speed_multiplier: f32,
    pub substeps: u32,
}

impl Default for TimelineState {
    fn default() -> Self {
        Self {
            is_playing: false,
            speed_multiplier: 1.0,
            substeps: 4,
        }
    }
}

/// Draw a circular transport button using Painter shapes.
/// Returns true if clicked.
fn transport_button(
    ui: &mut Ui,
    draw_icon: &dyn Fn(&egui::Painter, egui::Pos2, f32, egui::Color32),
    color: egui::Color32,
    tooltip: &str,
) -> bool {
    let size = Vec2::splat(theme::TRANSPORT_BUTTON_SIZE);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let painter = ui.painter();
    let center = rect.center();
    let r = size.x * 0.5;

    // Background circle
    let bg = if response.hovered() {
        theme::BG_WIDGET.gamma_multiply(1.2)
    } else {
        theme::BG_WIDGET
    };
    painter.circle_filled(center, r, bg);

    if response.hovered() {
        painter.circle_stroke(center, r, Stroke::new(1.0, color.gamma_multiply(0.5)));
    }

    // Draw icon
    let icon_color = if response.hovered() { color } else { color.gamma_multiply(0.7) };
    draw_icon(painter, center, r * 0.45, icon_color);

    let clicked = response.clicked();
    if response.hovered() {
        response.on_hover_text(tooltip);
    }

    clicked
}

/// Draw a play triangle or pause bars.
fn draw_play_pause(painter: &egui::Painter, center: egui::Pos2, s: f32, color: egui::Color32, is_playing: bool) {
    if is_playing {
        // Pause: two vertical bars
        let bar_w = s * 0.35;
        let gap = s * 0.3;
        painter.rect_filled(
            egui::Rect::from_center_size(center - egui::vec2(gap, 0.0), egui::vec2(bar_w, s * 2.0)),
            CornerRadius::same(1),
            color,
        );
        painter.rect_filled(
            egui::Rect::from_center_size(center + egui::vec2(gap, 0.0), egui::vec2(bar_w, s * 2.0)),
            CornerRadius::same(1),
            color,
        );
    } else {
        // Play: right-pointing triangle
        let points = vec![
            egui::pos2(center.x - s * 0.6, center.y - s),
            egui::pos2(center.x + s * 0.8, center.y),
            egui::pos2(center.x - s * 0.6, center.y + s),
        ];
        painter.add(egui::Shape::convex_polygon(
            points,
            color,
            Stroke::NONE,
        ));
    }
}

/// Draw step-forward icon (bar + triangle).
fn draw_step(painter: &egui::Painter, center: egui::Pos2, s: f32, color: egui::Color32) {
    // Small triangle
    let points = vec![
        egui::pos2(center.x - s * 0.7, center.y - s * 0.8),
        egui::pos2(center.x + s * 0.3, center.y),
        egui::pos2(center.x - s * 0.7, center.y + s * 0.8),
    ];
    painter.add(egui::Shape::convex_polygon(points, color, Stroke::NONE));
    // Vertical bar
    painter.rect_filled(
        egui::Rect::from_center_size(
            egui::pos2(center.x + s * 0.6, center.y),
            egui::vec2(s * 0.3, s * 1.6),
        ),
        CornerRadius::same(1),
        color,
    );
}

/// Draw reset icon (circular arrow approximation: a partial ring + arrow head).
fn draw_reset(painter: &egui::Painter, center: egui::Pos2, s: f32, color: egui::Color32) {
    // Draw a circular arc with an arrowhead using line segments
    let n = 10;
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let angle = std::f32::consts::PI * 0.3 + (i as f32 / (n - 1) as f32) * std::f32::consts::PI * 1.4;
        points.push(egui::pos2(
            center.x + angle.cos() * s * 1.2,
            center.y - angle.sin() * s * 1.2,
        ));
    }
    painter.add(egui::Shape::line(points, Stroke::new(1.5, color)));

    // Arrowhead at end
    let arrow_angle = std::f32::consts::PI * 0.3;
    let tip = egui::pos2(
        center.x + arrow_angle.cos() * s * 1.2,
        center.y - arrow_angle.sin() * s * 1.2,
    );
    let head = vec![
        egui::pos2(tip.x - s * 0.4, tip.y - s * 0.5),
        tip,
        egui::pos2(tip.x + s * 0.5, tip.y - s * 0.1),
    ];
    painter.add(egui::Shape::line(head, Stroke::new(1.5, color)));
}

/// Draw the transport controls. Returns true if reset was pressed.
pub fn draw_transport(ui: &mut Ui, state: &mut TimelineState) -> bool {
    let mut reset = false;

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;

        let is_playing = state.is_playing;
        let play_color = if is_playing { theme::TRANSPORT_PLAYING } else { theme::TEXT_PRIMARY };

        // Play/Pause
        if transport_button(
            ui,
            &|painter, center, s, color| draw_play_pause(painter, center, s, color, is_playing),
            play_color,
            if is_playing { "Pause" } else { "Play" },
        ) {
            state.is_playing = !state.is_playing;
        }

        // Step
        if transport_button(
            ui,
            &|painter, center, s, color| draw_step(painter, center, s, color),
            theme::TEXT_SECONDARY,
            "Step Forward",
        ) {
            state.is_playing = false;
            state.substeps = 1;
        }

        // Reset
        if transport_button(
            ui,
            &|painter, center, s, color| draw_reset(painter, center, s, color),
            theme::COLOR_ERASE.gamma_multiply(0.8),
            "Reset",
        ) {
            state.is_playing = false;
            reset = true;
        }
    });

    // Compact inline sliders
    ui.add_space(4.0);
    theme::section_frame().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Speed").size(10.0).color(theme::TEXT_DISABLED));
            ui.add(
                egui::Slider::new(&mut state.speed_multiplier, 0.1..=10.0)
                    .logarithmic(true)
                    .show_value(true),
            );
        });
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Substeps").size(10.0).color(theme::TEXT_DISABLED));
            ui.add(
                egui::Slider::new(&mut state.substeps, 1..=20)
                    .show_value(true),
            );
        });
    });

    reset
}
