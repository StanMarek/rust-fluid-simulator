use egui::Ui;

/// Scene selection index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenePreset {
    DamBreak,
    DoubleEmitter,
}

impl ScenePreset {
    pub fn label(&self) -> &'static str {
        match self {
            ScenePreset::DamBreak => "Dam Break",
            ScenePreset::DoubleEmitter => "Double Emitter",
        }
    }
}

/// Draw the scene panel.
/// Returns Some(preset) if the user selected a scene to load.
pub fn draw_scene_panel(ui: &mut Ui, current: &mut ScenePreset) -> Option<ScenePreset> {
    ui.heading("Scene");
    ui.separator();

    let mut load = None;

    let presets = [ScenePreset::DamBreak, ScenePreset::DoubleEmitter];
    for preset in &presets {
        let selected = *current == *preset;
        if ui.selectable_label(selected, preset.label()).clicked() {
            *current = *preset;
            load = Some(*preset);
        }
    }

    load
}
