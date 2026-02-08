use egui::Ui;
use renderer::color_map::ColorMapType;
use sim_core::scene::SceneDescription;

use crate::theme;

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

    const ALL: [ScenePreset; 2] = [ScenePreset::DamBreak, ScenePreset::DoubleEmitter];
}

fn color_map_label(cm: ColorMapType) -> &'static str {
    match cm {
        ColorMapType::Water => "Water",
        ColorMapType::Viridis => "Viridis",
        ColorMapType::Plasma => "Plasma",
        ColorMapType::Coolwarm => "Coolwarm",
    }
}

const ALL_COLOR_MAPS: [ColorMapType; 4] = [
    ColorMapType::Water,
    ColorMapType::Viridis,
    ColorMapType::Plasma,
    ColorMapType::Coolwarm,
];

/// What the scene panel wants the app to do.
pub enum SceneAction {
    LoadPreset(ScenePreset),
    LoadScene(SceneDescription),
}

/// Draw the scene & display panel.
/// Returns Some action if the user selected a scene to load.
pub fn draw_scene_panel(
    ui: &mut Ui,
    current: &mut ScenePreset,
    color_map: &mut ColorMapType,
    load_error: &mut Option<String>,
) -> Option<SceneAction> {
    let mut action = None;

    let scene_id = ui.make_persistent_id("scene_display");
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), scene_id, true)
        .show_header(ui, |ui| {
            ui.label(theme::section_heading("Scene & Display"));
        })
        .body(|ui| {
            theme::section_frame().show(ui, |ui| {
                // Scene preset ComboBox
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Scene").size(11.0).color(theme::TEXT_SECONDARY));
                    egui::ComboBox::from_id_salt("scene_combo")
                        .selected_text(current.label())
                        .show_ui(ui, |ui| {
                            for preset in &ScenePreset::ALL {
                                let selected = *current == *preset;
                                if ui.selectable_label(selected, preset.label()).clicked() {
                                    *current = *preset;
                                    action = Some(SceneAction::LoadPreset(*preset));
                                }
                            }
                        });
                });

                ui.add_space(4.0);

                // Load from file button
                if ui.button("Load File...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("JSON Scene", &["json"])
                        .pick_file()
                    {
                        match std::fs::read_to_string(&path) {
                            Ok(contents) => {
                                match serde_json::from_str::<SceneDescription>(&contents) {
                                    Ok(scene) => {
                                        *load_error = None;
                                        action = Some(SceneAction::LoadScene(scene));
                                    }
                                    Err(e) => {
                                        *load_error = Some(format!("Invalid JSON: {e}"));
                                    }
                                }
                            }
                            Err(e) => {
                                *load_error = Some(format!("Read error: {e}"));
                            }
                        }
                    }
                }

                // Show error if any
                if let Some(err) = load_error {
                    ui.colored_label(
                        theme::COLOR_ERASE,
                        egui::RichText::new(err.as_str()).size(10.0),
                    );
                }

                ui.add_space(4.0);

                // Color map ComboBox
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Colors").size(11.0).color(theme::TEXT_SECONDARY));
                    egui::ComboBox::from_id_salt("colormap_combo")
                        .selected_text(color_map_label(*color_map))
                        .show_ui(ui, |ui| {
                            for cm in &ALL_COLOR_MAPS {
                                let selected = *color_map == *cm;
                                if ui.selectable_label(selected, color_map_label(*cm)).clicked() {
                                    *color_map = *cm;
                                }
                            }
                        });
                });
            });
        });

    action
}
