use egui::Ui;

use crate::interaction::{InteractionState, Tool};

/// Draw the toolbar panel for tool selection.
pub fn draw_toolbar(ui: &mut Ui, interaction: &mut InteractionState) {
    ui.heading("Tools");
    ui.separator();

    let tools = [Tool::Emit, Tool::Drag, Tool::Erase, Tool::Obstacle];

    for tool in &tools {
        let selected = interaction.active_tool == *tool;
        if ui.selectable_label(selected, tool.label()).clicked() {
            interaction.active_tool = *tool;
        }
    }

    ui.separator();
    ui.label("Tool Settings");

    ui.add(
        egui::Slider::new(&mut interaction.tool_radius, 0.01..=0.2)
            .text("Radius")
            .logarithmic(true),
    );

    if interaction.active_tool == Tool::Emit {
        ui.add(egui::Slider::new(&mut interaction.tool_particle_count, 1..=500).text("Count"));
    }
}
