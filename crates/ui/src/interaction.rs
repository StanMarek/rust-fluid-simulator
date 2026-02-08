use renderer::camera::Camera2D;

/// Active tool for mouse interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    Emit,
    Drag,
    Erase,
    Obstacle,
}

impl Tool {
    pub fn label(&self) -> &'static str {
        match self {
            Tool::Emit => "Emit",
            Tool::Drag => "Drag",
            Tool::Erase => "Erase",
            Tool::Obstacle => "Obstacle",
        }
    }
}

/// Interaction state for the viewport.
pub struct InteractionState {
    pub active_tool: Tool,
    pub tool_radius: f32,
    pub tool_particle_count: usize,
    pub is_dragging: bool,
    pub last_mouse_world: Option<(f32, f32)>,
}

impl Default for InteractionState {
    fn default() -> Self {
        Self {
            active_tool: Tool::Emit,
            tool_radius: 0.05,
            tool_particle_count: 50,
            is_dragging: false,
            last_mouse_world: None,
        }
    }
}

impl InteractionState {
    /// Convert screen coordinates to world coordinates using the camera.
    pub fn screen_to_world(camera: &Camera2D, screen_x: f32, screen_y: f32) -> (f32, f32) {
        let world = camera.screen_to_world(screen_x, screen_y);
        (world.x, world.y)
    }
}
