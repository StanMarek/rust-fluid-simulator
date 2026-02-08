use std::time::Instant;

use common::{Dim2, Dimension, SimConfig};
use renderer::camera::Camera2D;
use renderer::color_map::{ColorMap, ColorMapType};
use renderer::particle_renderer::{ParticleInstance, ParticleRenderer};
use sim_core::particle::properties::DEFAULT_PARTICLE_RADIUS;
use sim_core::scene::SceneDescription;
use sim_core::Simulation;

use crate::interaction::{InteractionState, Tool};
use crate::panels::scene::ScenePreset;
use crate::panels::timeline::TimelineState;
use crate::panels::viewport::ViewportConfig;
use crate::theme;

/// Main application state. Orchestrates simulation, rendering, and UI.
pub struct FluidSimApp {
    // Simulation
    simulation: Simulation<Dim2>,

    // Rendering (initialized on first frame when wgpu is available)
    #[allow(dead_code)] // Used in later phases for wgpu rendering
    particle_renderer: Option<ParticleRenderer>,
    camera: Camera2D,
    #[allow(dead_code)] // Used in later phases for wgpu rendering
    viewport_config: ViewportConfig,
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
}

impl FluidSimApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let config = SimConfig::default();
        let simulation = Simulation::<Dim2>::new(config);

        Self {
            simulation,
            particle_renderer: None,
            camera: Camera2D::new(800.0, 600.0),
            viewport_config: ViewportConfig::default(),
            color_map_type: ColorMapType::Water,
            interaction: InteractionState::default(),
            timeline: TimelineState::default(),
            scene_preset: ScenePreset::DamBreak,
            last_frame_time: Instant::now(),
            fps: 0.0,
            frame_count: 0,
            needs_initial_load: true,
        }
    }

    /// Load a scene preset.
    fn load_scene(&mut self, preset: ScenePreset) {
        let scene = match preset {
            ScenePreset::DamBreak => SceneDescription::dam_break(),
            ScenePreset::DoubleEmitter => SceneDescription::double_emitter(),
        };
        self.simulation.load_scene(&scene);
    }

    /// Build particle instances for rendering from current simulation state.
    fn build_instances(&self) -> Vec<ParticleInstance> {
        let particles = &self.simulation.particles;
        let n = particles.len();
        let mut instances = Vec::with_capacity(n);

        // Compute velocity magnitude range for color mapping
        let mut max_vel = 1.0_f32;
        for i in 0..n {
            let vel = &particles.velocities[i];
            let speed = Dim2::magnitude(vel);
            if speed > max_vel {
                max_vel = speed;
            }
        }

        for i in 0..n {
            let pos = &particles.positions[i];
            let vel = &particles.velocities[i];
            let speed = Dim2::magnitude(vel);
            let t = (speed / max_vel).clamp(0.0, 1.0);
            let color = ColorMap::map(t, self.color_map_type);

            instances.push(ParticleInstance {
                world_pos: [Dim2::component(pos, 0), Dim2::component(pos, 1)],
                color,
                radius: DEFAULT_PARTICLE_RADIUS,
            });
        }

        instances
    }

    /// Handle viewport mouse interaction.
    fn handle_viewport_input(&mut self, ctx: &egui::Context) {
        // Get the central panel rect (viewport area)
        let input = ctx.input(|i| {
            (
                i.pointer.hover_pos(),
                i.pointer.primary_pressed(),
                i.pointer.primary_down(),
                i.pointer.middle_down(),
                i.pointer.button_released(egui::PointerButton::Primary),
                i.smooth_scroll_delta.y,
                i.pointer.delta(),
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

        // Tool interactions with left mouse button
        if let Some(pos) = hover_pos {
            let (wx, wy) = InteractionState::screen_to_world(&self.camera, pos.x, pos.y);

            if primary_pressed {
                self.interaction.is_dragging = true;
                self.interaction.last_mouse_world = Some((wx, wy));

                match self.interaction.active_tool {
                    Tool::Emit => {
                        self.simulation.spawn_particles_at(
                            wx,
                            wy,
                            self.interaction.tool_radius,
                            self.interaction.tool_particle_count,
                        );
                    }
                    Tool::Erase => {
                        self.simulation
                            .erase_particles_at(wx, wy, self.interaction.tool_radius);
                    }
                    Tool::Drag | Tool::Obstacle => {
                        // Drag: apply force each frame while held (handled in primary_down)
                    }
                }
            }

            if primary_down && self.interaction.is_dragging {
                match self.interaction.active_tool {
                    Tool::Emit => {
                        // Continuous emit while dragging
                        if let Some((lx, ly)) = self.interaction.last_mouse_world {
                            let dx = wx - lx;
                            let dy = wy - ly;
                            if dx * dx + dy * dy > (self.interaction.tool_radius * 0.5).powi(2) {
                                self.simulation.spawn_particles_at(
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
                        self.simulation
                            .erase_particles_at(wx, wy, self.interaction.tool_radius);
                    }
                    Tool::Drag => {
                        // Apply force to nearby particles
                        if let Some((lx, ly)) = self.interaction.last_mouse_world {
                            let fx = (wx - lx) * 500.0;
                            let fy = (wy - ly) * 500.0;
                            self.apply_drag_force(wx, wy, fx, fy);
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

    /// Apply a drag force to particles near a position.
    fn apply_drag_force(&mut self, x: f32, y: f32, fx: f32, fy: f32) {
        let radius = self.interaction.tool_radius;
        let radius_sq = radius * radius;
        let particles = &mut self.simulation.particles;

        for i in 0..particles.len() {
            let pos = &particles.positions[i];
            let px = Dim2::component(pos, 0);
            let py = Dim2::component(pos, 1);
            let dx = px - x;
            let dy = py - y;
            let dist_sq = dx * dx + dy * dy;

            if dist_sq < radius_sq {
                let factor = 1.0 - (dist_sq / radius_sq).sqrt();
                let force = Dim2::from_slice(&[fx * factor, fy * factor]);
                particles.velocities[i] += force;
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
        self.fps = self.fps * 0.95 + (1.0 / dt.max(0.001)) * 0.05; // Exponential moving average
        self.frame_count += 1;

        // Run simulation steps
        if self.timeline.is_playing {
            let substeps = self.timeline.substeps;
            for _ in 0..substeps {
                self.simulation.step();
            }
        }

        // Handle viewport input
        self.handle_viewport_input(ctx);

        // === UI Layout ===

        // Left panel: tools + properties
        egui::SidePanel::left("left_panel")
            .default_width(theme::SIDE_PANEL_WIDTH)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    crate::panels::toolbar::draw_toolbar(ui, &mut self.interaction);
                    ui.separator();
                    crate::panels::properties::draw_properties(
                        ui,
                        &mut self.simulation.config,
                        self.simulation.particles.len(),
                        self.fps,
                    );
                });
            });

        // Right panel: scene + timeline
        egui::SidePanel::right("right_panel")
            .default_width(theme::SIDE_PANEL_WIDTH)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if let Some(preset) =
                        crate::panels::scene::draw_scene_panel(ui, &mut self.scene_preset)
                    {
                        self.load_scene(preset);
                    }

                    ui.separator();

                    let reset = crate::panels::timeline::draw_timeline(
                        ui,
                        &mut self.timeline,
                        self.simulation.time,
                        self.simulation.step_count,
                    );

                    if reset {
                        self.load_scene(self.scene_preset);
                    }

                    ui.separator();
                    ui.label("Color Map");
                    let maps = [
                        (ColorMapType::Water, "Water"),
                        (ColorMapType::Viridis, "Viridis"),
                        (ColorMapType::Plasma, "Plasma"),
                        (ColorMapType::Coolwarm, "Coolwarm"),
                    ];
                    for (map_type, label) in &maps {
                        let selected = self.color_map_type == *map_type;
                        if ui.selectable_label(selected, *label).clicked() {
                            self.color_map_type = *map_type;
                        }
                    }
                });
            });

        // Central panel: viewport (egui-painted particles for Phase 1)
        egui::CentralPanel::default().show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            self.camera.set_viewport_size(rect.width(), rect.height());

            // Draw particles using egui painter (Phase 1 approach)
            // In later phases, this will be replaced with wgpu rendering
            let painter = ui.painter_at(rect);

            // Draw background
            painter.rect_filled(
                rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(13, 13, 20, 255),
            );

            // Draw domain boundary
            let domain_min_screen = self.world_to_screen(
                self.simulation.config.domain_min[0],
                self.simulation.config.domain_min[1],
                &rect,
            );
            let domain_max_screen = self.world_to_screen(
                self.simulation.config.domain_max[0],
                self.simulation.config.domain_max[1],
                &rect,
            );
            painter.rect_stroke(
                egui::Rect::from_min_max(
                    egui::pos2(domain_min_screen.0, domain_max_screen.1),
                    egui::pos2(domain_max_screen.0, domain_min_screen.1),
                ),
                0.0,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 80, 100)),
                egui::StrokeKind::Outside,
            );

            // Draw particles
            let instances = self.build_instances();
            for inst in &instances {
                let screen = self.world_to_screen(inst.world_pos[0], inst.world_pos[1], &rect);
                let radius_pixels = inst.radius * self.camera.zoom * rect.height();
                let radius_pixels = radius_pixels.max(1.5); // Minimum visible size

                let color = egui::Color32::from_rgb(
                    (inst.color[0] * 255.0) as u8,
                    (inst.color[1] * 255.0) as u8,
                    (inst.color[2] * 255.0) as u8,
                );

                painter.circle_filled(egui::pos2(screen.0, screen.1), radius_pixels, color);
            }

            // Draw tool cursor
            if let Some(pos) = ctx.input(|i| i.pointer.hover_pos()) {
                if rect.contains(pos) {
                    let tool_radius_pixels =
                        self.interaction.tool_radius * self.camera.zoom * rect.height();
                    painter.circle_stroke(
                        pos,
                        tool_radius_pixels,
                        egui::Stroke::new(
                            1.0,
                            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 100),
                        ),
                    );
                }
            }
        });

        // Request continuous repaint when simulation is running
        if self.timeline.is_playing {
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
