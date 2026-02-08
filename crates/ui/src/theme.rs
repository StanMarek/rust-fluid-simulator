use egui::{
    Color32, CornerRadius, FontId, Frame, Margin, Stroke, TextStyle, Visuals,
    style::WidgetVisuals,
};

use crate::interaction::Tool;

// ── Color palette: 4-tier depth system ──────────────────────────────────────

pub const BG_BASE: Color32 = Color32::from_rgb(18, 18, 22);
pub const BG_PANEL: Color32 = Color32::from_rgb(28, 28, 34);
pub const BG_SECTION: Color32 = Color32::from_rgb(36, 36, 44);
pub const BG_WIDGET: Color32 = Color32::from_rgb(46, 46, 56);

// ── Accent colors ───────────────────────────────────────────────────────────

pub const ACCENT_PRIMARY: Color32 = Color32::from_rgb(56, 142, 224);
pub const ACCENT_HOVER: Color32 = Color32::from_rgb(72, 158, 240);
pub const ACCENT_ACTIVE: Color32 = Color32::from_rgb(40, 120, 200);

// ── Text colors ─────────────────────────────────────────────────────────────

pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(220, 222, 228);
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(140, 144, 156);
pub const TEXT_DISABLED: Color32 = Color32::from_rgb(80, 82, 92);
pub const TEXT_ACCENT: Color32 = Color32::from_rgb(100, 170, 240);

// ── Borders ─────────────────────────────────────────────────────────────────

pub const BORDER_SUBTLE: Color32 = Color32::from_rgb(48, 48, 58);
pub const BORDER_WIDGET: Color32 = Color32::from_rgb(58, 58, 70);

// ── Tool-specific colors ────────────────────────────────────────────────────

pub const COLOR_EMIT: Color32 = Color32::from_rgb(80, 200, 120);
pub const COLOR_DRAG: Color32 = Color32::from_rgb(100, 160, 240);
pub const COLOR_ERASE: Color32 = Color32::from_rgb(240, 90, 90);
pub const COLOR_OBSTACLE: Color32 = Color32::from_rgb(200, 170, 80);

pub const TRANSPORT_PLAYING: Color32 = Color32::from_rgb(60, 180, 100);

pub const BG_STATUS: Color32 = Color32::from_rgb(22, 22, 28);

// ── Layout constants ────────────────────────────────────────────────────────

pub const SIDE_PANEL_WIDTH: f32 = 280.0;
pub const ITEM_SPACING: f32 = 4.0;
pub const SECTION_SPACING: f32 = 8.0;
pub const SECTION_INNER_MARGIN: f32 = 8.0;
pub const STATUS_BAR_HEIGHT: f32 = 24.0;
pub const TOOLBAR_BUTTON_SIZE: f32 = 36.0;
pub const TRANSPORT_BUTTON_SIZE: f32 = 28.0;

// ── Theme application ───────────────────────────────────────────────────────

pub fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    // Text styles
    style.text_styles = [
        (TextStyle::Heading, FontId::proportional(14.0)),
        (TextStyle::Body, FontId::proportional(12.0)),
        (TextStyle::Button, FontId::proportional(12.0)),
        (TextStyle::Small, FontId::proportional(10.0)),
        (TextStyle::Monospace, FontId::monospace(11.0)),
    ]
    .into();

    // Spacing
    style.spacing.item_spacing = egui::vec2(ITEM_SPACING, ITEM_SPACING);
    style.spacing.button_padding = egui::vec2(6.0, 3.0);
    style.spacing.slider_width = 140.0;
    style.spacing.scroll.bar_width = 6.0;

    let rounding = CornerRadius::same(4);
    let small_rounding = CornerRadius::same(3);

    // Widget visuals
    let mut visuals = Visuals::dark();

    visuals.panel_fill = BG_PANEL;
    visuals.window_fill = BG_PANEL;
    visuals.faint_bg_color = BG_SECTION;
    visuals.extreme_bg_color = BG_BASE;

    // Selection
    visuals.selection.bg_fill = ACCENT_PRIMARY.gamma_multiply(0.3);
    visuals.selection.stroke = Stroke::new(1.0, ACCENT_PRIMARY);

    // Noninteractive (labels, static text)
    visuals.widgets.noninteractive = WidgetVisuals {
        bg_fill: Color32::TRANSPARENT,
        weak_bg_fill: Color32::TRANSPARENT,
        bg_stroke: Stroke::NONE,
        corner_radius: rounding,
        fg_stroke: Stroke::new(1.0, TEXT_PRIMARY),
        expansion: 0.0,
    };

    // Inactive (interactive but not hovered)
    visuals.widgets.inactive = WidgetVisuals {
        bg_fill: BG_WIDGET,
        weak_bg_fill: BG_WIDGET,
        bg_stroke: Stroke::new(1.0, BORDER_WIDGET),
        corner_radius: rounding,
        fg_stroke: Stroke::new(1.0, TEXT_SECONDARY),
        expansion: 0.0,
    };

    // Hovered
    visuals.widgets.hovered = WidgetVisuals {
        bg_fill: BG_WIDGET.gamma_multiply(1.2),
        weak_bg_fill: BG_WIDGET.gamma_multiply(1.2),
        bg_stroke: Stroke::new(1.0, ACCENT_HOVER),
        corner_radius: rounding,
        fg_stroke: Stroke::new(1.0, TEXT_PRIMARY),
        expansion: 1.0,
    };

    // Active (pressed)
    visuals.widgets.active = WidgetVisuals {
        bg_fill: ACCENT_ACTIVE,
        weak_bg_fill: ACCENT_ACTIVE,
        bg_stroke: Stroke::new(1.0, ACCENT_PRIMARY),
        corner_radius: small_rounding,
        fg_stroke: Stroke::new(1.0, TEXT_PRIMARY),
        expansion: 0.0,
    };

    // Open (e.g. combo box open)
    visuals.widgets.open = WidgetVisuals {
        bg_fill: BG_SECTION,
        weak_bg_fill: BG_SECTION,
        bg_stroke: Stroke::new(1.0, ACCENT_PRIMARY),
        corner_radius: rounding,
        fg_stroke: Stroke::new(1.0, TEXT_PRIMARY),
        expansion: 0.0,
    };

    style.visuals = visuals;
    ctx.set_style(style);
}

// ── Helper functions ────────────────────────────────────────────────────────

/// Frame for collapsible section bodies.
pub fn section_frame() -> Frame {
    Frame::NONE
        .fill(BG_SECTION)
        .stroke(Stroke::new(1.0, BORDER_SUBTLE))
        .inner_margin(Margin::same(SECTION_INNER_MARGIN as i8))
        .corner_radius(CornerRadius::same(4))
}

/// Frame for the bottom status bar.
pub fn status_bar_frame() -> Frame {
    Frame::NONE
        .fill(BG_STATUS)
        .stroke(Stroke::new(1.0, BORDER_SUBTLE))
        .inner_margin(Margin::symmetric(12, 4))
}

/// Frame for the side panel.
pub fn side_panel_frame() -> Frame {
    Frame::NONE
        .fill(BG_PANEL)
        .inner_margin(Margin::same(8))
        .stroke(Stroke::new(1.0, BORDER_SUBTLE))
}

/// Section heading text: uppercase, small, secondary color.
pub fn section_heading(text: &str) -> egui::RichText {
    egui::RichText::new(text.to_uppercase())
        .size(11.0)
        .color(TEXT_SECONDARY)
        .strong()
}

/// Monospace accent-colored value text.
pub fn value_text(text: &str) -> egui::RichText {
    egui::RichText::new(text)
        .monospace()
        .color(TEXT_ACCENT)
        .size(11.0)
}

/// Get the color associated with a tool.
pub fn tool_color(tool: Tool) -> Color32 {
    match tool {
        Tool::Emit => COLOR_EMIT,
        Tool::Drag => COLOR_DRAG,
        Tool::Erase => COLOR_ERASE,
        Tool::Obstacle => COLOR_OBSTACLE,
    }
}

/// Get a Unicode/ASCII icon for a tool.
pub fn tool_icon(tool: Tool) -> &'static str {
    match tool {
        Tool::Emit => "+",
        Tool::Drag => "~",
        Tool::Erase => "x",
        Tool::Obstacle => "#",
    }
}
