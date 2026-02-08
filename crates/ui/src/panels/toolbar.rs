use egui::{CornerRadius, Sense, Stroke, Ui, Vec2};

use crate::interaction::{InteractionState, Tool};
use crate::theme;

/// Draw a single tool button. Returns true if clicked.
fn tool_button(ui: &mut Ui, tool: Tool, is_active: bool) -> bool {
    let size = Vec2::splat(theme::TOOLBAR_BUTTON_SIZE);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let painter = ui.painter();
    let color = theme::tool_color(tool);
    let icon = theme::tool_icon(tool);
    let label = tool.label();

    if is_active {
        // Active: tinted background + colored border
        painter.rect_filled(rect, CornerRadius::same(4), color.gamma_multiply(0.25));
        painter.rect_stroke(rect, CornerRadius::same(4), Stroke::new(1.5, color), egui::StrokeKind::Inside);
    } else if response.hovered() {
        painter.rect_filled(rect, CornerRadius::same(4), theme::BG_WIDGET);
        painter.rect_stroke(rect, CornerRadius::same(4), Stroke::new(1.0, theme::BORDER_SUBTLE), egui::StrokeKind::Inside);
    }

    // Icon (large, centered upper portion)
    let icon_color = if is_active { color } else if response.hovered() { theme::TEXT_PRIMARY } else { theme::TEXT_SECONDARY };
    painter.text(
        rect.center() - egui::vec2(0.0, 4.0),
        egui::Align2::CENTER_CENTER,
        icon,
        egui::FontId::monospace(14.0),
        icon_color,
    );

    // Tiny label below icon
    painter.text(
        egui::pos2(rect.center().x, rect.max.y - 5.0),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(8.0),
        if is_active { color } else { theme::TEXT_DISABLED },
    );

    response.clicked()
}

/// Draw the toolbar panel for tool selection.
pub fn draw_toolbar(ui: &mut Ui, interaction: &mut InteractionState) {
    ui.label(theme::section_heading("Tools"));
    ui.add_space(4.0);

    // 2x2 grid
    let tools = [
        [Tool::Emit, Tool::Drag],
        [Tool::Erase, Tool::Obstacle],
    ];

    for row in &tools {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for &tool in row {
                if tool_button(ui, tool, interaction.active_tool == tool) {
                    interaction.active_tool = tool;
                }
            }
        });
    }

    ui.add_space(4.0);

    // Contextual tool settings
    theme::section_frame().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Radius").size(11.0).color(theme::TEXT_SECONDARY));
            ui.add(
                egui::Slider::new(&mut interaction.tool_radius, 0.01..=0.2)
                    .logarithmic(true)
                    .show_value(true),
            );
        });

        if interaction.active_tool == Tool::Emit {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Count").size(11.0).color(theme::TEXT_SECONDARY));
                ui.add(
                    egui::Slider::new(&mut interaction.tool_particle_count, 1..=500)
                        .show_value(true),
                );
            });
        }
    });
}
