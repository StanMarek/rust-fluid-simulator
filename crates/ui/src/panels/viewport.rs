// Future: wgpu rendering surface integrated with egui.
// For Phase 1, the main rendering happens in app.rs using eframe's built-in
// wgpu integration. This module will provide additional viewport interaction
// handling (tool mouse input over the viewport).

/// Viewport configuration.
pub struct ViewportConfig {
    /// Background color (dark blue-gray).
    pub clear_color: [f32; 4],
}

impl Default for ViewportConfig {
    fn default() -> Self {
        Self {
            clear_color: [0.05, 0.05, 0.08, 1.0],
        }
    }
}
