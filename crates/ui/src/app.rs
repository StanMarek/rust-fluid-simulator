use std::sync::Arc;
use std::time::Instant;

use common::{Dim2, Dimension, SimConfig};
use egui_wgpu::wgpu;
use renderer::camera::Camera2D;
use renderer::color_map::{ColorMap, ColorMapType};
use renderer::particle_renderer::{ParticleInstance, ParticleRenderer};
use sim_core::obstacle::Obstacle;
use sim_core::particle::properties::DEFAULT_PARTICLE_RADIUS;
use sim_core::scene::SceneDescription;

use crate::backend::SimulationBackend;
use crate::interaction::{InteractionState, Tool};
use crate::panels::scene::{SceneAction, ScenePreset};
use crate::panels::timeline::TimelineState;
use crate::theme;

/// Main application state. Orchestrates simulation, rendering, and UI.
pub struct FluidSimApp {
    // Simulation backend (CPU or GPU)
    backend: SimulationBackend,

    // Rendering
    camera: Camera2D,
    color_map_type: ColorMapType,

    // UI state
    interaction: InteractionState,
    timeline: TimelineState,
    scene_preset: ScenePreset,

    // Performance tracking
    last_frame_time: Instant,
    fps: f32,
    frame_count: u64,

    // Flag for initial scene load
    needs_initial_load: bool,

    // Viewport rect from previous frame (used to gate pointer input)
    viewport_rect: egui::Rect,

    // Scene file loading error (displayed in scene panel)
    scene_load_error: Option<String>,
}

impl FluidSimApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let config = SimConfig::default();
        let backend = SimulationBackend::new_cpu(config);

        // Initialize the wgpu particle renderer and store in callback resources.
        if let Some(render_state) = &cc.wgpu_render_state {
            let renderer = ParticleRenderer::new(&render_state.device, render_state.target_format);
            render_state
                .renderer
                .write()
                .callback_resources
                .insert(renderer);
        }

        Self {
            backend,
            camera: Camera2D::new(800.0, 600.0),
            color_map_type: ColorMapType::Water,
            interaction: InteractionState::default(),
            timeline: TimelineState::default(),
            scene_preset: ScenePreset::DamBreak,
            last_frame_time: Instant::now(),
            fps: 0.0,
            frame_count: 0,
            needs_initial_load: true,
            viewport_rect: egui::Rect::NOTHING,
            scene_load_error: None,
        }
    }

    /// Load a scene preset.
    fn load_scene(&mut self, preset: ScenePreset) {
        let scene = match preset {
            ScenePreset::DamBreak => SceneDescription::dam_break(),
            ScenePreset::DoubleEmitter => SceneDescription::double_emitter(),
        };
        self.backend.load_scene(&scene);
    }

    /// Switch between CPU and GPU backends, reloading the current scene.
    fn switch_backend(&mut self, use_gpu: bool) {
        let config = self.backend.config().clone();
        if use_gpu {
            match SimulationBackend::new_gpu(config) {
                Some(mut new_backend) => {
                    let scene = match self.scene_preset {
                        ScenePreset::DamBreak => SceneDescription::dam_break(),
                        ScenePreset::DoubleEmitter => SceneDescription::double_emitter(),
                    };
                    new_backend.load_scene(&scene);
                    self.backend = new_backend;
                    log::info!("Switched to GPU backend");
                }
                None => {
                    log::warn!("GPU backend not available");
                }
            }
        } else {
            let mut new_backend = SimulationBackend::new_cpu(config);
            let scene = match self.scene_preset {
                ScenePreset::DamBreak => SceneDescription::dam_break(),
                ScenePreset::DoubleEmitter => SceneDescription::double_emitter(),
            };
            new_backend.load_scene(&scene);
            self.backend = new_backend;
            log::info!("Switched to CPU backend");
        }
    }

    /// Build particle instances for rendering from current simulation state.
    fn build_instances(&self) -> Vec<ParticleInstance> {
        let (positions, velocities) = self.backend.particle_data_for_rendering();
        let n = positions.len();
        let mut instances = Vec::with_capacity(n);

        // Compute velocity magnitude range for color mapping.
        let mut max_vel = 1.0_f32;
        for vel in &velocities {
            let speed = (vel[0] * vel[0] + vel[1] * vel[1]).sqrt();
            if speed > max_vel {
                max_vel = speed;
            }
        }

        for i in 0..n {
            let speed =
                (velocities[i][0] * velocities[i][0] + velocities[i][1] * velocities[i][1]).sqrt();
            let t = (speed / max_vel).clamp(0.0, 1.0);
            let color = ColorMap::map(t, self.color_map_type);

            instances.push(ParticleInstance {
                world_pos: positions[i],
                color,
                radius: DEFAULT_PARTICLE_RADIUS,
            });
        }

        instances
    }

    /// Handle viewport mouse interaction.
    fn handle_viewport_input(&mut self, ctx: &egui::Context) {
        let input = ctx.input(|i| {
            (
                i.pointer.hover_pos(),
                i.pointer.primary_pressed(),
                i.pointer.primary_down(),
                i.pointer.middle_down(),
                i.pointer.button_released(egui::PointerButton::Primary),
                i.smooth_scroll_delta.y,
                i.pointer.delta(),
                i.pointer.secondary_pressed(),
            )
        });

        let (
            hover_pos,
            primary_pressed,
            primary_down,
            middle_down,
            primary_released,
            scroll_y,
            delta,
            secondary_pressed,
        ) = input;

        // Zoom with scroll wheel
        if scroll_y != 0.0 {
            let factor = if scroll_y > 0.0 { 1.1 } else { 1.0 / 1.1 };
            self.camera.zoom_by(factor);
        }

        // Pan with middle mouse button
        if middle_down {
            self.camera.pan(delta.x, delta.y);
        }

        // Tool interactions with left mouse button (only within viewport and domain)
        if let Some(pos) = hover_pos.filter(|p| self.viewport_rect.contains(*p)) {
            // Offset by viewport origin so camera.screen_to_world gets local coords
            let local_x = pos.x - self.viewport_rect.min.x;
            let local_y = pos.y - self.viewport_rect.min.y;
            let (wx, wy) = InteractionState::screen_to_world(&self.camera, local_x, local_y);
            let domain = self.backend.config();
            let in_domain = wx >= domain.domain_min[0]
                && wx <= domain.domain_max[0]
                && wy >= domain.domain_min[1]
                && wy <= domain.domain_max[1];

            if !in_domain {
                // Outside domain: release any active drag but don't process tools
                if primary_released {
                    self.interaction.is_dragging = false;
                    self.interaction.last_mouse_world = None;
                }
                return;
            }

            // Right-click removes obstacles (when obstacle tool is active)
            if secondary_pressed && self.interaction.active_tool == Tool::Obstacle {
                self.remove_obstacle_at(wx, wy);
            }

            if primary_pressed {
                self.interaction.is_dragging = true;
                self.interaction.last_mouse_world = Some((wx, wy));

                match self.interaction.active_tool {
                    Tool::Emit => {
                        self.backend.spawn_particles_at(
                            wx,
                            wy,
                            self.interaction.tool_radius,
                            self.interaction.tool_particle_count,
                        );
                    }
                    Tool::Erase => {
                        self.backend
                            .erase_particles_at(wx, wy, self.interaction.tool_radius);
                    }
                    Tool::Obstacle => {
                        let center = Dim2::from_slice(&[wx, wy]);
                        self.backend.push_obstacle(Obstacle::Circle {
                            center,
                            radius: self.interaction.tool_radius,
                        });
                    }
                    Tool::Drag => {}
                }
            }

            if primary_down && self.interaction.is_dragging {
                match self.interaction.active_tool {
                    Tool::Emit => {
                        if let Some((lx, ly)) = self.interaction.last_mouse_world {
                            let dx = wx - lx;
                            let dy = wy - ly;
                            if dx * dx + dy * dy > (self.interaction.tool_radius * 0.5).powi(2) {
                                self.backend.spawn_particles_at(
                                    wx,
                                    wy,
                                    self.interaction.tool_radius,
                                    self.interaction.tool_particle_count / 4,
                                );
                                self.interaction.last_mouse_world = Some((wx, wy));
                            }
                        }
                    }
                    Tool::Erase => {
                        self.backend
                            .erase_particles_at(wx, wy, self.interaction.tool_radius);
                    }
                    Tool::Drag => {
                        if let Some((lx, ly)) = self.interaction.last_mouse_world {
                            let fx = (wx - lx) * 500.0;
                            let fy = (wy - ly) * 500.0;
                            self.backend.apply_force_at(
                                wx,
                                wy,
                                fx,
                                fy,
                                self.interaction.tool_radius,
                            );
                        }
                        self.interaction.last_mouse_world = Some((wx, wy));
                    }
                    Tool::Obstacle => {}
                }
            }

            if primary_released {
                self.interaction.is_dragging = false;
                self.interaction.last_mouse_world = None;
            }
        }
    }

    /// Remove the obstacle closest to the given world position (if within tool radius).
    fn remove_obstacle_at(&mut self, x: f32, y: f32) {
        let point = Dim2::from_slice(&[x, y]);
        let mut best_idx = None;
        let mut best_dist = f32::INFINITY;
        for (i, obs) in self.backend.obstacles().iter().enumerate() {
            let d = obs.sdf(&point).abs();
            if d < best_dist {
                best_dist = d;
                best_idx = Some(i);
            }
        }
        if let Some(idx) = best_idx {
            if best_dist < self.interaction.tool_radius * 2.0 {
                self.backend.remove_obstacle(idx);
            }
        }
    }
}

impl eframe::App for FluidSimApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Apply theme on first frame
        if self.frame_count == 0 {
            theme::apply_theme(ctx);
        }

        // Load initial scene
        if self.needs_initial_load {
            self.load_scene(self.scene_preset);
            self.needs_initial_load = false;
        }

        // FPS calculation
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        self.fps = self.fps * 0.95 + (1.0 / dt.max(0.001)) * 0.05;
        self.frame_count += 1;

        // Run simulation steps
        if self.timeline.step_once {
            self.backend.step();
            self.timeline.step_once = false;
        }
        if self.timeline.is_playing {
            let scaled = (self.timeline.substeps as f32 * self.timeline.speed_multiplier).round() as u32;
            let substeps = scaled.clamp(1, 100);
            for _ in 0..substeps {
                self.backend.step();
            }
        }

        // Handle viewport input
        self.handle_viewport_input(ctx);

        // === UI Layout ===

        // Bottom status bar
        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(theme::STATUS_BAR_HEIGHT)
            .frame(theme::status_bar_frame())
            .show(ctx, |ui| {
                crate::panels::status::draw_status_bar(
                    ui,
                    self.backend.particle_count(),
                    self.fps,
                    self.backend.time(),
                    self.backend.step_count(),
                    self.timeline.is_playing,
                    self.backend.backend_name(),
                );
            });

        // Left panel: all controls in one sidebar
        egui::SidePanel::left("left_panel")
            .exact_width(theme::SIDE_PANEL_WIDTH)
            .frame(theme::side_panel_frame())
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    // Transport controls (always visible at top)
                    let reset = crate::panels::timeline::draw_transport(
                        ui,
                        &mut self.timeline,
                    );
                    if reset {
                        self.load_scene(self.scene_preset);
                    }

                    ui.add_space(theme::SECTION_SPACING);

                    // Tools section
                    crate::panels::toolbar::draw_toolbar(ui, &mut self.interaction);

                    ui.add_space(theme::SECTION_SPACING);

                    // Properties sections (4 collapsible sub-sections)
                    crate::panels::properties::draw_properties(
                        ui,
                        self.backend.config_mut(),
                    );

                    ui.add_space(theme::SECTION_SPACING);

                    // Scene & Display section
                    if let Some(action) = crate::panels::scene::draw_scene_panel(
                        ui,
                        &mut self.scene_preset,
                        &mut self.color_map_type,
                        &mut self.scene_load_error,
                        self.backend.is_gpu(),
                    ) {
                        match action {
                            SceneAction::LoadPreset(preset) => {
                                self.load_scene(preset);
                            }
                            SceneAction::LoadScene(scene) => {
                                self.backend.load_scene(&scene);
                            }
                            SceneAction::SwitchBackend(use_gpu) => {
                                self.switch_backend(use_gpu);
                            }
                        }
                    }
                });
            });

        // Central panel: viewport with wgpu instanced particle rendering.
        let instances = self.build_instances();
        let camera_uniform = renderer::camera::CameraUniform::from_camera(&self.camera);

        egui::CentralPanel::default().show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            self.viewport_rect = rect;
            self.camera.set_viewport_size(rect.width(), rect.height());

            let painter = ui.painter_at(rect);

            // Draw background
            painter.rect_filled(rect, 0.0, theme::BG_BASE);

            // Draw domain boundary with subtle styling
            let config = self.backend.config();
            let domain_min_screen = self.world_to_screen(
                config.domain_min[0],
                config.domain_min[1],
                &rect,
            );
            let domain_max_screen = self.world_to_screen(
                config.domain_max[0],
                config.domain_max[1],
                &rect,
            );
            let domain_rect = egui::Rect::from_min_max(
                egui::pos2(domain_min_screen.0, domain_max_screen.1),
                egui::pos2(domain_max_screen.0, domain_min_screen.1),
            );

            // Inner dark stroke for depth
            painter.rect_stroke(
                domain_rect,
                egui::CornerRadius::same(2),
                egui::Stroke::new(2.0, egui::Color32::from_rgba_unmultiplied(10, 10, 14, 120)),
                egui::StrokeKind::Outside,
            );
            // Outer visible stroke
            painter.rect_stroke(
                domain_rect,
                egui::CornerRadius::same(2),
                egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 62, 80)),
                egui::StrokeKind::Outside,
            );

            // Corner accent L-shapes
            let corner_len = 12.0;
            let cc = egui::Color32::from_rgba_unmultiplied(80, 140, 220, 80);
            let cs = egui::Stroke::new(1.5, cc);
            // Top-left
            painter.line_segment([domain_rect.left_top(), domain_rect.left_top() + egui::vec2(corner_len, 0.0)], cs);
            painter.line_segment([domain_rect.left_top(), domain_rect.left_top() + egui::vec2(0.0, corner_len)], cs);
            // Top-right
            painter.line_segment([domain_rect.right_top(), domain_rect.right_top() + egui::vec2(-corner_len, 0.0)], cs);
            painter.line_segment([domain_rect.right_top(), domain_rect.right_top() + egui::vec2(0.0, corner_len)], cs);
            // Bottom-left
            painter.line_segment([domain_rect.left_bottom(), domain_rect.left_bottom() + egui::vec2(corner_len, 0.0)], cs);
            painter.line_segment([domain_rect.left_bottom(), domain_rect.left_bottom() + egui::vec2(0.0, -corner_len)], cs);
            // Bottom-right
            painter.line_segment([domain_rect.right_bottom(), domain_rect.right_bottom() + egui::vec2(-corner_len, 0.0)], cs);
            painter.line_segment([domain_rect.right_bottom(), domain_rect.right_bottom() + egui::vec2(0.0, -corner_len)], cs);

            // Render particles via wgpu instanced draw callback.
            let cb = egui_wgpu::Callback::new_paint_callback(
                rect,
                ParticleDrawCallback {
                    instances: Arc::new(instances),
                    camera_uniform,
                },
            );
            painter.add(cb);

            // Draw obstacles
            let obs_fill = theme::COLOR_OBSTACLE.gamma_multiply(0.15);
            let obs_stroke = egui::Stroke::new(1.5, theme::COLOR_OBSTACLE.gamma_multiply(0.6));
            for obstacle in self.backend.obstacles() {
                match obstacle {
                    Obstacle::Circle { center, radius } => {
                        let (sx, sy) = self.world_to_screen(
                            Dim2::component(center, 0),
                            Dim2::component(center, 1),
                            &rect,
                        );
                        let r_pixels = radius * self.camera.zoom * rect.height();
                        painter.circle_filled(egui::pos2(sx, sy), r_pixels, obs_fill);
                        painter.circle_stroke(egui::pos2(sx, sy), r_pixels, obs_stroke);
                    }
                    Obstacle::Box { min, max } => {
                        let (sx_min, sy_min) = self.world_to_screen(
                            Dim2::component(min, 0),
                            Dim2::component(min, 1),
                            &rect,
                        );
                        let (sx_max, sy_max) = self.world_to_screen(
                            Dim2::component(max, 0),
                            Dim2::component(max, 1),
                            &rect,
                        );
                        let obs_rect = egui::Rect::from_min_max(
                            egui::pos2(sx_min, sy_max),
                            egui::pos2(sx_max, sy_min),
                        );
                        painter.rect_filled(obs_rect, 0.0, obs_fill);
                        painter.rect_stroke(obs_rect, egui::CornerRadius::ZERO, obs_stroke, egui::StrokeKind::Outside);
                    }
                }
            }

            // Draw tool cursor with tool-specific color (only inside domain box)
            if let Some(pos) = ctx.input(|i| i.pointer.hover_pos()) {
                if domain_rect.contains(pos) {
                    let tool_radius_pixels =
                        self.interaction.tool_radius * self.camera.zoom * rect.height();
                    let cursor_color = theme::tool_color(self.interaction.active_tool);

                    // Outer circle
                    painter.circle_stroke(
                        pos,
                        tool_radius_pixels,
                        egui::Stroke::new(
                            1.0,
                            cursor_color.gamma_multiply(0.5),
                        ),
                    );
                    // Center dot
                    painter.circle_filled(pos, 2.0, cursor_color.gamma_multiply(0.7));
                }
            }
        });

        // Request continuous repaint when simulation is running or step was requested
        if self.timeline.is_playing || self.timeline.step_once {
            ctx.request_repaint();
        }
    }
}

impl FluidSimApp {
    /// Convert world coordinates to screen coordinates within a given rect.
    fn world_to_screen(&self, wx: f32, wy: f32, rect: &egui::Rect) -> (f32, f32) {
        let half_width = rect.width() / (2.0 * self.camera.zoom * rect.height());
        let half_height = 0.5 / self.camera.zoom;

        let ndc_x = (wx - self.camera.center.x) / half_width;
        let ndc_y = (wy - self.camera.center.y) / half_height;

        let screen_x = rect.center().x + ndc_x * rect.width() * 0.5;
        let screen_y = rect.center().y - ndc_y * rect.height() * 0.5;

        (screen_x, screen_y)
    }
}

/// Callback that uploads instance data and issues the wgpu instanced draw call.
struct ParticleDrawCallback {
    instances: Arc<Vec<ParticleInstance>>,
    camera_uniform: renderer::camera::CameraUniform,
}

impl egui_wgpu::CallbackTrait for ParticleDrawCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let renderer: &mut ParticleRenderer = callback_resources.get_mut().unwrap();
        renderer.update_instances(device, queue, &self.instances);
        queue.write_buffer(
            renderer.camera_buffer(),
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let renderer: &ParticleRenderer = callback_resources.get().unwrap();
        renderer.render(render_pass);
    }
}
