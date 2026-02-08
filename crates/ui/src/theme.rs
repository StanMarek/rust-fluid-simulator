use egui::{Color32, Visuals};

/// Apply the fluid simulator UI theme.
pub fn apply_theme(ctx: &egui::Context) {
    let mut visuals = Visuals::dark();

    // Slightly softer background
    visuals.panel_fill = Color32::from_rgb(30, 30, 35);
    visuals.window_fill = Color32::from_rgb(35, 35, 40);

    // Accent color for widgets
    visuals.widgets.active.bg_fill = Color32::from_rgb(60, 120, 200);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(50, 100, 180);

    ctx.set_visuals(visuals);
}

/// Standard panel width in pixels.
pub const SIDE_PANEL_WIDTH: f32 = 250.0;

/// Standard spacing between UI elements.
pub const ITEM_SPACING: f32 = 6.0;
