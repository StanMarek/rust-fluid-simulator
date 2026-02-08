use glam::{Mat4, Vec2};

/// 2D camera with pan and zoom.
/// Later extensible to 3D orbit camera.
pub struct Camera2D {
    /// Camera center position in world space.
    pub center: Vec2,
    /// Zoom level (pixels per world unit at scale 1.0).
    pub zoom: f32,
    /// Viewport size in pixels.
    pub viewport_size: Vec2,
}

impl Camera2D {
    pub fn new(viewport_width: f32, viewport_height: f32) -> Self {
        Self {
            center: Vec2::new(0.5, 0.5),
            zoom: 1.0,
            viewport_size: Vec2::new(viewport_width, viewport_height),
        }
    }

    /// Compute the view-projection matrix for rendering.
    /// Maps world coordinates to clip space [-1, 1].
    pub fn view_projection(&self) -> Mat4 {
        let half_width = self.viewport_size.x / (2.0 * self.zoom * self.viewport_size.y);
        let half_height = 0.5 / self.zoom;

        Mat4::orthographic_rh(
            self.center.x - half_width,
            self.center.x + half_width,
            self.center.y - half_height,
            self.center.y + half_height,
            -1.0,
            1.0,
        )
    }

    /// Pan the camera by a delta in screen pixels.
    pub fn pan(&mut self, dx_pixels: f32, dy_pixels: f32) {
        let scale = 1.0 / (self.zoom * self.viewport_size.y);
        self.center.x -= dx_pixels * scale;
        self.center.y += dy_pixels * scale; // Y is flipped in screen space
    }

    /// Zoom in/out centered on the current camera center.
    pub fn zoom_by(&mut self, factor: f32) {
        self.zoom *= factor;
        self.zoom = self.zoom.clamp(0.1, 100.0);
    }

    /// Convert screen coordinates to world coordinates.
    pub fn screen_to_world(&self, screen_x: f32, screen_y: f32) -> Vec2 {
        let scale = 1.0 / (self.zoom * self.viewport_size.y);
        let ndc_x = (screen_x / self.viewport_size.x) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_y / self.viewport_size.y) * 2.0;

        let half_width = self.viewport_size.x * scale * 0.5;
        let half_height = 0.5 / self.zoom;

        Vec2::new(
            self.center.x + ndc_x * half_width,
            self.center.y + ndc_y * half_height,
        )
    }

    /// Update viewport size.
    pub fn set_viewport_size(&mut self, width: f32, height: f32) {
        self.viewport_size = Vec2::new(width, height);
    }

    /// Get the view-projection matrix as a column-major [f32; 16] array for GPU upload.
    pub fn view_projection_array(&self) -> [f32; 16] {
        self.view_projection().to_cols_array()
    }
}

/// Uniform buffer data for the camera, uploaded to the GPU.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [f32; 16],
}

impl CameraUniform {
    pub fn from_camera(camera: &Camera2D) -> Self {
        Self {
            view_proj: camera.view_projection_array(),
        }
    }
}
