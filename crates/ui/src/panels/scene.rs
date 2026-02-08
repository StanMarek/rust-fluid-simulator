use egui::Ui;
use renderer::color_map::ColorMapType;

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

/// Draw the scene & display panel.
/// Returns Some(preset) if the user selected a scene to load.
pub fn draw_scene_panel(
    ui: &mut Ui,
    current: &mut ScenePreset,
    color_map: &mut ColorMapType,
) -> Option<ScenePreset> {
    let mut load = None;

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
                                    load = Some(*preset);
                                }
                            }
                        });
                });

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

    load
}
