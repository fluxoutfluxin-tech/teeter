//! A small realtime 3-band EQ used to make the synth and the live system audio
//! sit in complementary frequency bands so they mix musically instead of
//! clashing.
//!
//! Each band is a biquad: low-shelf (bass), peaking (mid) and high-shelf
//! (treble). Per-channel state, updated with the RBJ cookbook coefficients
//! whenever the sample rate or a gain changes.

/// A single biquad section (Direct Form I), enough for a shelf/peak.
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    fn new() -> Self {
        Self { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0, x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0 }
    }

    #[inline]
    fn tick(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// A low-shelf / peaking / high-shelf EQ set for one channel.
#[derive(Clone, Copy, PartialEq)]
pub struct EqGains {
    /// Bass gain in dB (-12..+12).
    pub bass_db: f64,
    /// Mid gain in dB (-12..+12).
    pub mid_db: f64,
    /// Treble gain in dB (-12..+12).
    pub treble_db: f64,
}

impl Default for EqGains {
    fn default() -> Self {
        Self { bass_db: 0.0, mid_db: 0.0, treble_db: 0.0 }
    }
}

/// One channel's EQ: three cascaded biquads.
pub struct EqChannel {
    bass: Biquad,
    mid: Biquad,
    treble: Biquad,
    gains: EqGains,
    sample_rate: f64,
}

impl EqChannel {
    /// Create a flat EQ at the given sample rate.
    pub fn new(sample_rate: u32) -> Self {
        let mut c = Self {
            bass: Biquad::new(),
            mid: Biquad::new(),
            treble: Biquad::new(),
            gains: EqGains::default(),
            sample_rate: sample_rate as f64,
        };
        c.recompute();
        c
    }

    /// Update a band gain in dB and recompute that coefficient set.
    pub fn set_gains(&mut self, gains: EqGains) {
        self.gains = gains;
        self.recompute();
    }

    fn recompute(&mut self) {
        let fs = self.sample_rate.max(1.0);
        low_shelf(&mut self.bass, fs, 120.0, 0.9, self.gains.bass_db);
        peaking(&mut self.mid, fs, 1000.0, 1.0, self.gains.mid_db);
        high_shelf(&mut self.treble, fs, 4500.0, 0.9, self.gains.treble_db);
    }

    #[inline]
    pub fn tick(&mut self, x: f64) -> f64 {
        self.treble.tick(self.mid.tick(self.bass.tick(x)))
    }
}

/// RBJ cookbook low-shelf filter.
fn low_shelf(b: &mut Biquad, fs: f64, f0: f64, q: f64, gain_db: f64) {
    let a = 10.0f64.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let sqrt_a = 2.0 * a.sqrt() * alpha;
    let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + sqrt_a;
    set_biquad(
        b,
        a * ((a + 1.0) - (a - 1.0) * cos_w0 + sqrt_a),
        2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0),
        a * ((a + 1.0) - (a - 1.0) * cos_w0 - sqrt_a),
        -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0),
        (a + 1.0) - (a - 1.0) * cos_w0 - sqrt_a,
        a0,
    );
}

/// RBJ cookbook peaking (parametric) filter.
fn peaking(b: &mut Biquad, fs: f64, f0: f64, q: f64, gain_db: f64) {
    let a = 10.0f64.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let a0 = 1.0 + alpha / a;
    set_biquad(
        b,
        1.0 + alpha * a,
        -2.0 * cos_w0,
        1.0 - alpha * a,
        -2.0 * cos_w0,
        1.0 - alpha / a,
        a0,
    );
}

/// RBJ cookbook high-shelf filter.
fn high_shelf(b: &mut Biquad, fs: f64, f0: f64, q: f64, gain_db: f64) {
    let a = 10.0f64.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let sqrt_a = 2.0 * a.sqrt() * alpha;
    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + sqrt_a;
    set_biquad(
        b,
        a * ((a + 1.0) + (a - 1.0) * cos_w0 + sqrt_a),
        -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0),
        a * ((a + 1.0) + (a - 1.0) * cos_w0 - sqrt_a),
        2.0 * ((a - 1.0) - (a + 1.0) * cos_w0),
        (a + 1.0) - (a - 1.0) * cos_w0 - sqrt_a,
        a0,
    );
}

/// Assign normalized coefficients (divide by `a0`).
#[allow(clippy::too_many_arguments)]
fn set_biquad(
    b: &mut Biquad,
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    a0: f64,
) {
    let inv = 1.0 / a0;
    b.b0 = b0 * inv;
    b.b1 = b1 * inv;
    b.b2 = b2 * inv;
    b.a1 = a1 * inv;
    b.a2 = a2 * inv;
}
