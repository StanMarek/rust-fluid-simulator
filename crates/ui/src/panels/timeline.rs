use egui::Ui;

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

/// Draw the timeline panel.
/// Returns true if "reset" was pressed.
pub fn draw_timeline(
    ui: &mut Ui,
    state: &mut TimelineState,
    sim_time: f32,
    step_count: u64,
) -> bool {
    ui.heading("Timeline");
    ui.separator();

    let mut reset = false;

    ui.horizontal(|ui| {
        let play_label = if state.is_playing { "Pause" } else { "Play" };
        if ui.button(play_label).clicked() {
            state.is_playing = !state.is_playing;
        }
        if ui.button("Step").clicked() {
            state.is_playing = false;
            // Return value indicates a single step should occur — handled by app
            state.substeps = 1; // Will be reset after one frame
        }
        if ui.button("Reset").clicked() {
            state.is_playing = false;
            reset = true;
        }
    });

    ui.add(
        egui::Slider::new(&mut state.speed_multiplier, 0.1..=10.0)
            .text("Speed")
            .logarithmic(true),
    );

    ui.add(egui::Slider::new(&mut state.substeps, 1..=20).text("Substeps/frame"));

    ui.separator();
    ui.label(format!("Time: {:.4}s", sim_time));
    ui.label(format!("Steps: {}", step_count));

    reset
}
