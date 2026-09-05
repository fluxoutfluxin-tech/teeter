//! Engine state: the feedback-loop uniforms and audio feature data that are
//! pushed to the GPU each frame.
//!
//! Stage 1 keeps this as a self-contained state struct fed by the input layer
//! and (later) the audio analyzer. Stage 2 replaces the placeholder audio
//! features with a real WASAPI loopback + FFT analysis.

/// Feature/resonance bundle sent to the shader each frame.
#[derive(Clone, Copy)]
pub struct FrameUniforms {
    pub i_time: f32,
    pub bass: f32,
    pub mid: f32,
    pub treble: f32,
    pub beat: f32,
    pub zoom: f32,
    pub warp_x: f32,
    pub warp_y: f32,
    pub rotate: f32,
    pub dissolve: f32,
    pub palette: f32,
    pub preset_seed: f32,
}

impl Default for FrameUniforms {
    fn default() -> Self {
        Self {
            i_time: 0.0,
            bass: 0.0,
            mid: 0.0,
            treble: 0.0,
            beat: 0.0,
            zoom: 0.0,
            warp_x: 0.0,
            warp_y: 0.0,
            rotate: 0.0,
            dissolve: 0.0,
            palette: 0.0,
            preset_seed: 0.0,
        }
    }
}

/// A stable-per-preset seed derived from the preset index / random choice.
fn seed_for_preset(idx: u32) -> f32 {
    // wrapping_mul: idx grows unbounded as presets cycle, and idx * the Knuth
    // constant overflows u32 by idx>=2. Debug builds panic on plain `*`, so we
    // wrap to keep cycling forever without crashing.
    let h = (idx.wrapping_mul(2654435761u32) >> 8) as f32;
    (h / 16777216.0).fract()
}

pub struct EngineState {
    pub frame: u64,
    pub preset_index: u32,
    pub preset_seed: f32,
    /// Embellishment (wave/beam) amplitude driven by beat, decays each tick.
    pub beat_pulse: f32,
}

impl Default for EngineState {
    fn default() -> Self {
        let preset_index = 0;
        Self {
            frame: 0,
            preset_index,
            preset_seed: seed_for_preset(preset_index),
            beat_pulse: 0.0,
        }
    }
}

impl EngineState {
    /// Advance one logical frame. `dt_sec` is wall-clock time since start.
    pub fn tick(&mut self, time: f32) {
        self.frame += 1;
        self.beat_pulse *= 0.95;
        let _ = time;
    }

    /// Build the uniform block for the current frame from input + audio.
    pub fn uniforms(&self, input: &crate::input::InputState, audio: &crate::audio::AudioFrame) -> FrameUniforms {
        let t = self.frame as f32 * (1.0 / 60.0);
        let beat = self.beat_pulse.max(audio.beat);

        // --- Music-reactive "blend & jump with the music" -----------------
        // These derive continuous motion + beat-locked jumps from the detected
        // genre / energy / tempo, so the visuals shift character with the music.
        let (auto_zoom_speed, auto_warp, auto_dissolve, palette_drift) = self.genre_motion(audio, t);

        // Zoom: breathing by energy + a jump on every beat.
        let auto_zoom = auto_zoom_speed * (0.25 + 0.75 * audio.energy) + beat * 0.55;
        // Warp: genre-scaled traveling waves (x*cos + y*sin Lissajous).
        let auto_wx = auto_warp * (0.6 * (t * 0.9).sin() + 0.4 * (t * 2.3).sin());
        let auto_wy = auto_warp * (0.6 * (t * 1.1).cos() + 0.4 * (t * 2.7).cos());
        let auto_rot = auto_warp * 0.3 * (t * 0.5).sin();
        let auto_palette = (t * palette_drift).rem_euclid(1.0);

        // Blend: manual input wins when the player actively moves it,
        // otherwise the music drives. 1.0 = full manual, 0 = full auto.
        let man = self.manual_weight(input);
        let zoom = lerp(auto_zoom as f64, input.zoom as f64, man).clamp(-2.0, 2.0) as f32;
        let warp_x = lerp(auto_wx as f64, input.warp_x as f64, man).clamp(-1.0, 1.0) as f32;
        let warp_y = lerp(auto_wy as f64, input.warp_y as f64, man).clamp(-1.0, 1.0) as f32;
        let rotate = lerp(auto_rot as f64, input.rotate as f64, man).clamp(-1.0, 1.0) as f32;
        let dissolve = (input.dissolve as f32)
            .max(auto_dissolve + beat * 0.25 + 0.05)
            .clamp(0.0, 1.0);
        let palette = (input.palette as f32 + auto_palette).rem_euclid(1.0);

        FrameUniforms {
            i_time: t,
            bass: audio.bass,
            mid: audio.mid,
            treble: audio.treble,
            beat,
            zoom,
            warp_x,
            warp_y,
            rotate,
            dissolve,
            palette,
            preset_seed: self.preset_seed,
        }
    }

    /// Genre-driven motion tuning: (zoom breath speed, warp amplitude,
    /// base dissolve, palette drift rate).
    fn genre_motion(&self, a: &crate::audio::AudioFrame, _t: f32) -> (f32, f32, f32, f32) {
        use crate::audio::Genre;
        match a.genre {
            Genre::Ambient => (0.25, 0.15, 0.12, 0.0006),
            Genre::Bass => (0.55, 0.45, 0.2, 0.0018),
            Genre::Rhythmic => (0.8, 0.6, 0.3, 0.0012),
            Genre::Melodic => (0.45, 0.4, 0.24, 0.0010),
            Genre::Inactive => (0.3, 0.2, 0.12, 0.0008),
        }
    }

    /// How much manual control takes over (0..1). Lifts toward 1 the more the
    /// player is actively moving the sticks, so idle = music drives.
    fn manual_weight(&self, input: &crate::input::InputState) -> f64 {
        let activity = input.zoom.abs().max(input.warp_x.abs()).max(input.warp_y.abs())
            .max(input.rotate.abs()) as f64;
        (activity * 4.0).clamp(0.0, 1.0)
    }

    /// Cycle to the next preset (re-seeds the feedback look).
    pub fn next_preset(&mut self) {
        self.preset_index = self.preset_index.wrapping_add(1);
        self.preset_seed = seed_for_preset(self.preset_index);
        self.beat_pulse = 1.0;
    }
}

/// Linear blend: value at `t=0.0` is `a`, at `t=1.0` is `b`.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0)
}
