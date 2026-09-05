//! Input: gamepad (gilrs) + touchscreen, folded into a compact navigation
//! state that drives the engine uniforms. Fullscreen immersive, no cursor.
//!
//! Mapping (ROG Ally controller, Xbox layout):
//!   - Left stick  Y -> warp_y,  X -> warp_x
//!   - Right stick X -> rotate,    Y -> zoom
//!   - LB / RB / D-pad L/R / triggers -> palette shift (manual color cycle)
//!   - A (South)      -> next preset
//!   - B (East)       -> next warp shader
//!   - Touch: swipe = warp tilt, pinch = zoom, tap = next preset (Stage 3)

use gilrs::Gilrs;

/// Compact per-frame control values in [-1, 1].
pub struct InputState {
    pub warp_x: f32,
    pub warp_y: f32,
    pub rotate: f32,
    pub zoom: f32,
    pub dissolve: f32,
    pub palette: f32,
    pub next_preset_edge: bool,
    pub next_shader_edge: bool,
    pub toggle_fullscreen_edge: bool,
    /// Rising edges requesting more/less live system-audio blend (D-pad up/down).
    pub live_mix_inc_edge: bool,
    pub live_mix_dec_edge: bool,
    /// Synth master gain axis 0..1 (right trigger analog).
    pub master_axis: f32,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            warp_x: 0.0,
            warp_y: 0.0,
            rotate: 0.0,
            zoom: 0.0,
            dissolve: 0.0,
            palette: 0.0,
            next_preset_edge: false,
            next_shader_edge: false,
            toggle_fullscreen_edge: false,
            live_mix_inc_edge: false,
            live_mix_dec_edge: false,
            master_axis: 0.0,        }
    }
}

/// Owns the gamepad backend so it persists across polls.
pub struct Controller {
    pub gilrs: Gilrs,
    pub state: InputState,
    // Rising-edge latches so a held trigger axis nudges the palette once.
    lt_was_high: bool,
    rt_was_high: bool,
}

impl Controller {
    pub fn new() -> Self {
        let gilrs = Gilrs::new().ok();
        if let Some(g) = &gilrs {
            log::info!("input: gilrs found {} connected pad(s)", g.gamepads().count());
        }
        Self {
            gilrs: gilrs.expect("failed to init gilrs (no gamepad backend)"),
            state: InputState::default(),
            lt_was_high: false,
            rt_was_high: false,
        }
    }

    pub fn poll(&mut self) {
        // Reset edge latches before reading new events.
        self.state.next_preset_edge = false;
        self.state.next_shader_edge = false;
        self.state.toggle_fullscreen_edge = false;
        self.state.live_mix_inc_edge = false;
        self.state.live_mix_dec_edge = false;

        while let Some(event) = self.gilrs.next_event() {
            use gilrs::EventType;
            match &event.event {
                EventType::ButtonPressed(button, _) => {
                    // Diagnostic: log every button press so we can see which
                    // variant ROG Ally reports for LB/RB.
                    log::info!("input: button pressed: {:?}", *button);
                    match *button {
                        gilrs::Button::South => self.state.next_preset_edge = true,
                        // B (East) cycles the warp shader; keeps A = preset.
                        gilrs::Button::East => self.state.next_shader_edge = true,
                        // Color cycle: accept LB/RB (both variants) AND D-pad
                        // L/R so there's always a reliable control on the Ally.
                        gilrs::Button::LeftTrigger
                        | gilrs::Button::LeftTrigger2
                        | gilrs::Button::DPadLeft => {
                            self.state.palette = (self.state.palette - 0.15).rem_euclid(1.0);
                            log::info!("input: palette -> {:.2}", self.state.palette);
                        }
                        gilrs::Button::RightTrigger
                        | gilrs::Button::RightTrigger2
                        | gilrs::Button::DPadRight => {
                            self.state.palette = (self.state.palette + 0.15).rem_euclid(1.0);
                            log::info!("input: palette -> {:.2}", self.state.palette);
                        }
                        // Start / Menu toggles fullscreen.
                        gilrs::Button::Start => self.state.toggle_fullscreen_edge = true,
                        // D-pad up/down adjust the live system-audio blend level.
                        gilrs::Button::DPadUp => self.state.live_mix_inc_edge = true,
                        gilrs::Button::DPadDown => self.state.live_mix_dec_edge = true,
                        _ => {}
                    }
                }
                EventType::AxisChanged(axis, value, _) => {
                    let v: f32 = (*value).into();
                    match axis {
                        gilrs::Axis::LeftStickX => self.state.warp_x = v,
                        gilrs::Axis::LeftStickY => self.state.warp_y = v,
                        gilrs::Axis::RightStickX => self.state.rotate = v,
                        gilrs::Axis::RightStickY => self.state.zoom = v,
                        // Analog triggers also nudge the color cycle so manual
                        // control always works even if the bumper buttons don't
                        // register as buttons on this controller. Only nudge on
                        // a rising edge (pull) so a held trigger fires once.
                        gilrs::Axis::LeftZ => {
                            self.state.dissolve = v.clamp(0.0, 1.0);
                            let high = v > 0.5;
                            if high && !self.lt_was_high {
                                self.state.palette = (self.state.palette - 0.15).rem_euclid(1.0);
                                log::info!("input: palette -> {:.2}", self.state.palette);
                            }
                            self.lt_was_high = high;
                        }
                        gilrs::Axis::RightZ => {
                            self.state.dissolve = v.clamp(0.0, 1.0);
                            let high = v > 0.5;
                            if high && !self.rt_was_high {
                                self.state.palette = (self.state.palette + 0.15).rem_euclid(1.0);
                                log::info!("input: palette -> {:.2}", self.state.palette);
                            }
                            self.rt_was_high = high;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }
}

impl Default for Controller {
    fn default() -> Self {
        Self::new()
    }
}
