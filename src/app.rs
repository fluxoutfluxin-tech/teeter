//! Top-level application: owns the window, the Vulkan renderer, the audio
//! analysis feed, and input (gamepad + touch). Implements winit's
//! `ApplicationHandler` so the event loop drives everything.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Icon, Window, WindowId};

use crate::audio::AudioAnalyzer;
use crate::engine::EngineState;
use crate::input::Controller;
use crate::vulkan::Renderer;

/// 512x512 RGBA logo (generated from assets/logo.svg). Embedded so the window
/// and taskbar show the teeter mark without pulling in an image crate.
const LOGO_WIDTH: u32 = 512;
const LOGO_HEIGHT: u32 = 512;
const LOGO_RGBA: &[u8] = include_bytes!("../assets/logo.rgba");

fn window_icon() -> Icon {
    Icon::from_rgba(LOGO_RGBA.to_vec(), LOGO_WIDTH, LOGO_HEIGHT)
        .expect("in-bounds teeter icon")
}

pub struct App {
    pub window: Option<Arc<Window>>,
    pub renderer: Option<Renderer>,
    pub engine: EngineState,
    pub controller: Controller,
    pub audio: Box<dyn AudioAnalyzer>,
    /// Index into the trip-engine preset list; advanced alongside the visual
    /// preset so the synth and the shader change together.
    synth_preset: usize,
    /// Optional CLI boot targets (`--shader N` / `--preset N`).
    pub boot_shader: Option<usize>,
    pub boot_preset: Option<usize>,
    pub start: std::time::Instant,
    /// Last frame's elapsed time (for dt).
    last_elapsed: f32,
    /// Touch-down timestamp, used to distinguish a quick tap (next preset)
    /// from a hold (fullscreen toggle). `None` = no touch in progress.
    touch_down: Option<std::time::Instant>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            engine: EngineState::default(),
            controller: Controller::new(),
            audio: Box::new(crate::audio::SynthAudio::default()),
            synth_preset: 0,
            boot_shader: None,
            boot_preset: None,
            start: std::time::Instant::now(),
            last_elapsed: 0.0,
            touch_down: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("teeter")
            .with_resizable(true)
            .with_window_icon(Some(window_icon()));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );

        let renderer = Renderer::new(Arc::clone(&window)).expect("failed to create renderer");
        if let Err(e) = self.audio.start() {
            log::warn!("audio analyzer failed to start: {e}");
        }

        self.window = Some(window);
        self.renderer = Some(renderer);

        // Apply CLI boot targets (--shader N / --preset N).
        if let Some(s) = self.boot_shader.take() {
            if let Some(r) = &mut self.renderer {
                r.set_shader_index(s);
                self.update_title();
            }
        }
        if let Some(p) = self.boot_preset.take() {
            self.engine.set_preset(p as u32);
            let n = trip_engine::presets::NAMES.len();
            if n > 0 {
                self.synth_preset = (p as usize) % n;
                if let Some(h) = self.audio.handle() {
                    let preset = trip_engine::presets::at(self.synth_preset);
                    log::info!("app: boot preset -> {} ({})", preset.name, p);
                    h.request_preset(preset);
                }
            }
            self.update_title();
        }

        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.render(&self.engine, &self.controller.state, &self.audio.current_frame());
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::Resized(_) => {
                if let Some(renderer) = &mut self.renderer {
                    if let Some(w) = &self.window {
                        let _ = renderer.recreate_swapchain(w);
                    }
                }
            }
            WindowEvent::Touch(t) => self.on_touch(&t),
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::event::ElementState;
                use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
                if event.repeat {
                    return;
                }
                let pressed = event.state == ElementState::Pressed;
                if !pressed {
                    return;
                }
                let dir = match event.physical_key {
                    PhysicalKey::Code(KeyCode::BracketLeft) => -0.1,
                    PhysicalKey::Code(KeyCode::BracketRight) => 0.1,
                    PhysicalKey::Code(KeyCode::Comma) => -0.1,
                    PhysicalKey::Code(KeyCode::Period) => 0.1,
                    PhysicalKey::Code(KeyCode::ArrowLeft) => -0.1,
                    PhysicalKey::Code(KeyCode::ArrowRight) => 0.1,
                    _ => match event.logical_key {
                        Key::Named(NamedKey::ArrowLeft) => -0.1,
                        Key::Named(NamedKey::ArrowRight) => 0.1,
                        _ => 0.0,
                    },
                };
                if dir != 0.0 {
                    self.adjust_live_mix(dir);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.controller.poll();
        if self.controller.state.next_preset_edge {
            self.engine.next_preset();
            // Advance the trip-engine synth preset in lockstep so the sound
            // and the visuals change together.
            self.advance_synth_preset();
            self.update_title();
        }
        if self.controller.state.next_shader_edge {
            if let Some(renderer) = &mut self.renderer {
                renderer.next_shader();
            }
            // B3: re-seed so each warp lands on a fresh colour character.
            self.engine.reshuffle_seed();
            self.update_title();
        }
        if self.controller.state.fivecell_toggle_edge {
            if let Some(renderer) = &mut self.renderer {
                renderer.toggle_fivecell();
            }
        }
        if self.controller.state.toggle_fullscreen_edge {
            self.toggle_fullscreen();
        }
        if self.controller.state.live_mix_inc_edge {
            self.adjust_live_mix(0.1);
        }
        if self.controller.state.live_mix_dec_edge {
            self.adjust_live_mix(-0.1);
        }
        // Right trigger = synth master gain. Only poke while pulled so a
        // released trigger holds the last level instead of dropping to silence.
        let master = self.controller.state.master_axis;
        if master > 0.01 {
            if let Some(h) = self.audio.handle() {
                let mut ctl = h.snapshot();
                if (ctl.master - master as f64).abs() > 1e-4 {
                    ctl.master = master as f64;
                    h.set_params(ctl);
                    log::info!("app: master gain -> {:.2}", master);
                }
            }
        }

        // Auto presets are OFF — presets only change via the A button (See
        // next_preset_edge above). Disabled here so a preset stays put.
        let dt = self.start.elapsed().as_secs_f32() - self.last_elapsed;
        self.last_elapsed = self.start.elapsed().as_secs_f32();
        let _ = dt; // dt kept only to maintain the elapsed baseline
        self.engine.tick(self.start.elapsed().as_secs_f32());
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Drop the renderer (and thus the Vulkan device/surface) on the main
        // thread before the event loop finishes.
        self.renderer = None;
    }
}

impl App {
    /// Handle a touch event: a quick tap cycles the preset; a hold (~0.7s)
    /// toggles fullscreen.
    fn on_touch(&mut self, t: &winit::event::Touch) {
        use winit::event::TouchPhase;
        match t.phase {
            TouchPhase::Started => {
                self.touch_down = Some(std::time::Instant::now());
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                let Some(down) = self.touch_down.take() else { return };
                if down.elapsed().as_secs_f32() >= 0.7 {
                    // Long hold -> fullscreen toggle.
                    self.toggle_fullscreen();
                } else {
                    // Quick tap -> next preset (synth + visuals together).
                    self.engine.next_preset();
                    self.advance_synth_preset();
                    self.update_title();
                }
            }
            _ => {}
        }
    }

    /// Cycle the trip-engine synth to the next preset (cyclically).
    fn advance_synth_preset(&mut self) {
        if let Some(h) = self.audio.handle() {
            self.synth_preset = (self.synth_preset + 1) % trip_engine::presets::NAMES.len();
            let preset = trip_engine::presets::at(self.synth_preset);
            log::info!("app: synth preset -> {}", preset.name);
            h.request_preset(preset);
        }
    }

    /// Nudge how much of the default system audio (loopback) is blended into
    /// the synth, clamped to [0, 1]. Reads/writes through the audio handle so
    /// the realtime Blender picks it up each block.
    fn adjust_live_mix(&mut self, delta: f64) {
        let Some(h) = self.audio.handle() else { return };
        let mut ctl = h.snapshot();
        let new = (ctl.live_mix + delta).clamp(0.0, 1.0);
        if (new - ctl.live_mix).abs() < 1e-6 {
            return;
        }
        ctl.live_mix = new;
        h.set_params(ctl);
        log::info!("app: live system audio mix -> {:.2}", new);
    }

    /// Toggle between windowed and borderless-fullscreen.
    fn toggle_fullscreen(&mut self) {
        use winit::window::Fullscreen;
        if let Some(w) = &self.window {
            let full = w.fullscreen().is_some();
            if full {
                w.set_fullscreen(None);
            } else {
                let monitor = w.current_monitor().or_else(|| w.primary_monitor());
                w.set_fullscreen(Some(Fullscreen::Borderless(monitor)));
            }
            log::info!("app: fullscreen -> {}", !full);
            // The OS fires a Resized event when entering/leaving fullscreen,
            // which drives the swapchain rebuild. No manual recreate here.
            w.request_redraw();
        }
    }

    /// Reflect the active warp in the title bar (used on shader switches and
    /// preset changes so the current look is always visible at a glance).
    fn update_title(&mut self) {
        let Some(w) = &self.window else { return };
        let (idx, total) = self
            .renderer
            .as_ref()
            .map(|r| (r.shader_index() + 1, r.shader_count()))
            .unwrap_or((0, 0));
        let name = self
            .renderer
            .as_ref()
            .and_then(|r| r.warp_meta())
            .map(|(n, g)| format!(" — {n} ({g})"))
            .unwrap_or_default();
        let _ = w.set_title(&format!("teeter — warp {idx}/{total}{name}"));
    }
}
