//! Audio analysis driven by the trip-engine live synth.
//!
//! The app runs a single realtime pipeline: the trip-engine [`Synth`] plays
//! through the speakers on a cpal output thread while a WASAPI loopback stream
//! (via `trip_engine::Loopback`) captures whatever else the system plays. A
//! [`Blender`] EQ's both layers, mixes them, and tees the blended stereo into a
//! shared ring. The renderer calls [`AudioAnalyzer::current_frame`] each frame,
//! which runs a windowed FFT over that ring to produce the [`AudioFrame`] band
//! energies + a beat-onset pulse.
//!
//! The loops are deliberately decoupled: the capture thread never blocks on
//! the renderer and vice-versa, so neither can starve the other.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

pub mod eq;

/// Live synth control handle (trip-engine), re-exported for the analyzer trait.
pub use trip_engine::SynthHandle;

/// Detected musical character, used to drive how the visuals "move" with the
/// music so the look blends and jumps differently per genre.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum Genre {
    #[default]
    Inactive,
    /// Sparse / slow / wide — slow drifting visuals.
    Ambient,
    /// Strong low end, steady pulse — pumping zoom + bass-driven warp.
    Bass,
    /// Fast rhythmic / percussive — beat-locked jumps and dissolves.
    Rhythmic,
    /// Mid-forward melodic — flowing warp, gentle palette drift.
    Melodic,
}

/// Normalized per-frame audio features consumed by the engine/shaders.
#[derive(Clone, Copy, Default)]
pub struct AudioFrame {
    pub bass: f32,   // 0..1
    pub mid: f32,    // 0..1
    pub treble: f32, // 0..1
    pub beat: f32,   // 0..1 beat onset pulse
    /// Current detected genre (drives visual character).
    pub genre: Genre,
    /// Overall loudness 0..1 (smoothed).
    pub energy: f32,
    /// Rolling beat-frequency estimate 0..1 (0 = sparse/ambient, 1 = fast).
    pub tempo: f32,
    /// Sub-bass band (20-80 Hz), same 0..1 dB-mapped scale as `bass`.
    pub sub_bass: f32,
    /// Spectral centroid, 0 = deep/bass-heavy, 1 = bright/high-end.
    pub centroid: f32,
    /// Spectral crest factor (peak/mean of the spectrum), 0..1 peaking.
    pub crest: f32,
    /// Spectral-flux proxy: frame-to-frame band novelty 0..1 (rises on hits).
    pub flux: f32,
    /// Rolloff frequency (85% of energy below), normalized 0..1 by Nyquist.
    pub rolloff: f32,
}

/// A minimal capture/analyzer trait so the renderer never needs to know the
/// backend.
pub trait AudioAnalyzer {
    fn current_frame(&self) -> AudioFrame;
    fn start(&mut self) -> Result<(), String>;
    /// Optional live-synth control handle (only the synth backend provides it).
    fn handle(&self) -> Option<SynthHandle> {
        None
    }
}

/// Size of the FFT window (power of two). ~46ms at 44.1k.
const FFT_SIZE: usize = 2048;
/// How much history we keep (FFT window plus smoothing headroom).
const RING_MAX: usize = FFT_SIZE * 4;
/// Per-frame decay for the beat pulse.
const PULSE_DECAY: f32 = 0.945;
/// Attack time coefficient (how fast a band rises on a transient).
const ATTACK: f32 = 0.45;
/// Release time coefficient (how fast a band falls when energy drops).
const RELEASE: f32 = 0.06;

// dB mapping: mean magnitude (normalized) -> 0..1. Audio energy spans ~100 dB,
// so linear gain cannot span the range; map a window of dB to 0..1 instead.
const DB_FLOOR: f32 = -45.0; // mean magnitude dB mapped to 0
const DB_CEIL: f32 = -5.0; // mean magnitude dB mapped to 1
/// Guard against log(0).
const DB_EPS: f32 = 1e-9;

/// Rolling beat envelope shared between analyze and the idle fallback.
#[derive(Default)]
struct BeatState {
    baseline: f32,
    pulse: f32,
    /// Time (secs) of the last detected beat onset, for tempo estimation.
    last_beat: f32,
    /// Time (secs) of the previous beat onset, for the interval.
    prev_beat: f32,
    /// Rolling mean interpolated beat interval (secs).
    interval: f32,
}

/// Shared state between the capture thread and the render thread.
struct Shared {
    /// Recent mono samples, newest at the back.
    ring: Mutex<VecDeque<f32>>,
    /// Latest computed features (written on the render thread).
    frame: RwLock<AudioFrame>,
    /// Rolling beat envelope (written on the render thread).
    beat: Mutex<BeatState>,
    /// Sample rate reported by the capture endpoint.
    sample_rate: AtomicU32,
}
impl Shared {
    fn new() -> Self {
        Self {
            ring: Mutex::new(VecDeque::with_capacity(RING_MAX)),
            frame: RwLock::new(AudioFrame::default()),
            beat: Mutex::new(BeatState::default()),
            sample_rate: AtomicU32::new(48000),
        }
    }
}

/// Shared ring -> normalized [`AudioFrame`]. Uses the latest `FFT_SIZE`
/// samples, with idle shimmer before data arrives or during silence.
fn frame_from_shared(shared: &Arc<Shared>, idle: &Cell<f32>) -> AudioFrame {
    let frame = {
        let mut ring = shared.ring.lock().unwrap();
        let rate = shared.sample_rate.load(Ordering::Relaxed);

        // No captured samples yet -> keep an idle shimmer so the screen is
        // never a dead black while the stream warms up.
        if ring.is_empty() {
            return idle_frame(idle);
        }

        let prev = *shared.frame.read().unwrap();
        let frame = analyze(&mut ring, rate, &shared.beat, &prev);

        // Silence detection: if the window is essentially flat, fall back
        // to idle so the shader still has something non-zero to react to.
        let peak = ring.iter().rev().take(FFT_SIZE).fold(0.0f32, |m, s| m.max(s.abs()));
        if peak < 0.0005 {
            return idle_frame(idle);
        }

        // Trim to RING_MAX to bound memory.
        while ring.len() > RING_MAX {
            ring.pop_front();
        }
        frame
    };
    *shared.frame.write().unwrap() = frame;
    frame
}

/// Analyzer driven by the trip-engine live synth.
///
/// This is the "combined tripengine" audio source: instead of capturing the
/// system loopback, we run the trip-engine [`Synth`] on a realtime cpal output
/// thread (so the synths actually plays through the speakers) and tee its
/// stereo frames back into the same shared ring / FFT path the loopback
/// analyzer uses. The renderer keeps calling [`AudioAnalyzer::current_frame`]
/// and gets the synth's band energies + beat exactly as before.
pub struct SynthAudio {
    shared: Option<Arc<Shared>>,
    handle: Option<SynthHandle>,
    stop: Option<Arc<AtomicBool>>,
    audio_handle: Option<thread::JoinHandle<()>>,
    idle: Cell<f32>,
}

impl Default for SynthAudio {
    fn default() -> Self {
        Self { shared: None, handle: None, stop: None, audio_handle: None, idle: Cell::new(0.0) }
    }
}

impl AudioAnalyzer for SynthAudio {
    /// Cloneable handle the UI thread can use to switch presets / adjust the
    /// live synth controls. Only available after [`start`](AudioAnalyzer::start).
    fn handle(&self) -> Option<SynthHandle> {
        self.handle.clone()
    }

    fn start(&mut self) -> Result<(), String> {
        if self.shared.is_some() {
            return Ok(()); // already running
        }
        let shared = Arc::new(Shared::new());
        let cap = Arc::clone(&shared);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();

        // One synth handle stays on this (render/UI) thread for control pokes;
        // the audio thread keeps its own copy.
        let synth = trip_engine::Synth::new(trip_engine::presets::at(0));
        let handle = synth.handle();

        let audio_handle = thread::Builder::new()
            .name("tripengine-synth".to_string())
            .spawn(move || {
                let shared = cap; // owned Arc moved into this thread
                if let Err(e) = synth_output_thread(synth, stop_thread, shared) {
                    log::warn!("tripengine synth audio stopped: {e}");
                }
            })
            .map_err(|e| format!("spawn synth thread: {e}"))?;

        self.shared = Some(shared);
        self.handle = Some(handle);
        self.stop = Some(stop);
        self.audio_handle = Some(audio_handle);
        Ok(())
    }

    fn current_frame(&self) -> AudioFrame {
        let Some(shared) = &self.shared else {
            return idle_frame(&self.idle);
        };
        frame_from_shared(shared, &self.idle)
    }
}

/// Realtime synth-to-speaker thread. Opens the default output device, builds a
/// cpal stream, fills it from the trip-engine `Synth`, live-mixes the system
/// audio (WASAPI loopback) through a 3-band EQ so the synth sits musically
/// with whatever is playing, and tees the blended stereo into the shared ring
/// so the FFT analyzer sees the whole mix.
fn synth_output_thread(
    mut synth: trip_engine::Synth,
    stop: Arc<AtomicBool>,
    shared: Arc<Shared>,
) -> Result<(), String> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    // Flush denormals so the feedback DSP never spikes on the APU.
    trip_engine::cpu::set_flush_denormals();

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| "no default output device".to_string())?;
    log::info!("tripengine: output device: {}", device.name().map_err(|e| e.to_string())?);

    let supported = device.default_output_config().map_err(|e| e.to_string())?;
    let sample_format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    shared.sample_rate.store(config.sample_rate.0, Ordering::Relaxed);

    // Render the synth at the output device's actual rate.
    synth.set_rate(config.sample_rate.0);
    let synth_handle = synth.handle();

    // Start system-audio loopback capture for live blending. If it fails (no
    // render device), we fall back to synth-only without error.
    let mut loopback = trip_engine::Loopback::new();
    if let Err(e) = loopback.start() {
        log::warn!("tripengine: loopback capture unavailable; live blend disabled: {e}");
    }

    // Build a per-block blender that EQ's both layers and mixes them, teeing
    // the blended result into the analyzer ring internally.
    let mut blender = Blender::new(config.sample_rate.0, synth, loopback, synth_handle, shared);

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            device
                .build_output_stream(
                    &config,
                    move |data: &mut [f32], _| {
                        blender.fill_f32(data);
                    },
                    audio_error,
                    None,
                )
                .map_err(|e| e.to_string())?
        }
        cpal::SampleFormat::I16 => {
            device
                .build_output_stream(
                    &config,
                    move |data: &mut [i16], _| {
                        blender.fill_i16(data);
                    },
                    audio_error,
                    None,
                )
                .map_err(|e| e.to_string())?
        }
        cpal::SampleFormat::F64 => {
            device
                .build_output_stream(
                    &config,
                    move |data: &mut [f64], _| {
                        blender.fill_f64(data);
                    },
                    audio_error,
                    None,
                )
                .map_err(|e| e.to_string())?
        }
        other => return Err(format!("unsupported sample format {other:?}")),
    };

    stream.play().map_err(|e| e.to_string())?;
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    stream.pause().map_err(|e| e.to_string())?;
    Ok(())
}

/// Blends the synth with live system audio through a 3-band EQ. Owned by the
/// cpal callback; the blended result is teed into the analyzer ring each block
/// so the FFT analyzer sees the whole mix (system audio + synth), not just the
/// raw synth.
/// Makeup gain on the live system-audio layer: the loopback capture tends to
/// read well below the synth, so lift it to sit in the mix rather than whisper.
const LIVE_GAIN: f64 = 2.5;
struct Blender {
    synth: trip_engine::Synth,
    handle: SynthHandle,
    loopback: trip_engine::Loopback,
    ring: Arc<Shared>,
    /// Per-channel EQ for the synth layer (shaped to poke through the mix).
    eq_synth_l: eq::EqChannel,
    eq_synth_r: eq::EqChannel,
    /// Per-channel EQ for the live system audio layer.
    eq_live_l: eq::EqChannel,
    eq_live_r: eq::EqChannel,
    live: Vec<(f32, f32)>,
    mix: f64,
    /// Cached blended frames from the last fill.
    last: Vec<trip_engine::out::Frame>,
}

impl Blender {
    fn new(
        sample_rate: u32,
        synth: trip_engine::Synth,
        loopback: trip_engine::Loopback,
        handle: SynthHandle,
        ring: Arc<Shared>,
    ) -> Self {
        // Default complementary EQ: the synth keeps its body/mids, the system
        // audio's lows and highs come through, both shaped so they don't clash.
        let mut eq_synth_l = eq::EqChannel::new(sample_rate);
        let mut eq_synth_r = eq::EqChannel::new(sample_rate);
        eq_synth_l.set_gains(eq::EqGains { bass_db: -4.0, mid_db: 3.0, treble_db: 1.0 });
        eq_synth_r.set_gains(eq::EqGains { bass_db: -4.0, mid_db: 3.0, treble_db: 1.0 });
        let mut eq_live_l = eq::EqChannel::new(sample_rate);
        let mut eq_live_r = eq::EqChannel::new(sample_rate);
        eq_live_l.set_gains(eq::EqGains { bass_db: 4.0, mid_db: 2.0, treble_db: 3.0 });
        eq_live_r.set_gains(eq::EqGains { bass_db: 4.0, mid_db: 2.0, treble_db: 3.0 });

        // Blend system audio into the mix by default. If nothing has set
        // live_mix yet (the UI can poke it later), default to a healthy level
        // so the live/equalized blend actually happens.
        let mut ctl = handle.snapshot();
        if ctl.live_mix <= 0.0 {
            ctl.live_mix = 1.0;
            handle.set_params(ctl);
        }

        Self {
            synth,
            handle,
            loopback,
            ring,
            eq_synth_l,
            eq_synth_r,
            eq_live_l,
            eq_live_r,
            live: Vec::new(),
            mix: 0.0,
            last: Vec::new(),
        }
    }

    /// Current synth control snapshot (drives the live-mix level).
    fn snapshot_mix(&mut self) -> f64 {
        let ctl = self.handle.snapshot();
        self.mix = ctl.live_mix;
        self.mix
    }

    /// Fill an interleaved F32 output slice with the blended mix.
    fn fill_f32(&mut self, out: &mut [f32]) {
        let n = out.len() / 2;
        self.blend_block(n, |i, l, r| {
            out[i * 2] = l as f32;
            out[i * 2 + 1] = r as f32;
        });
    }

    /// Fill interleaved F64 output slice.
    fn fill_f64(&mut self, out: &mut [f64]) {
        let n = out.len() / 2;
        self.blend_block(n, |i, l, r| {
            out[i * 2] = l;
            out[i * 2 + 1] = r;
        });
    }

    /// Fill interleaved I16 output slice.
    fn fill_i16(&mut self, out: &mut [i16]) {
        let n = out.len() / 2;
        self.blend_block(n, |i, l, r| {
            out[i * 2] = (l.clamp(-1.0, 1.0) * 32767.0) as i16;
            out[i * 2 + 1] = (r.clamp(-1.0, 1.0) * 32767.0) as i16;
        });
    }

    /// Render `n` frames: synth -> EQ, system live -> EQ, blend by live-mix,
    /// tee the blended result into the analyzer ring, then write to `write`.
    fn blend_block(&mut self, n: usize, mut write: impl FnMut(usize, f64, f64)) {
        let mix = self.snapshot_mix();

        self.last.resize(n, trip_engine::out::Frame { l: 0.0, r: 0.0 });
        self.synth.fill(&mut self.last);

        // Pull one block of system audio (stereo) into `live`.
        self.live.resize(n, (0.0, 0.0));
        let has_live = self.loopback.read_block(&mut self.live);
        let mix = if has_live { mix } else { 0.0 };
        let mut peak = 0.0f64;

        // Apply to each frame: EQ synth, EQ live, blend, soft gate.
        let frames = std::mem::take(&mut self.last);
        for (i, fr) in frames.iter().enumerate() {
            let sl = self.eq_synth_l.tick(fr.l);
            let sr = self.eq_synth_r.tick(fr.r);
            let (ll, lr) = self.live[i];
            let ll = self.eq_live_l.tick(ll as f64) * LIVE_GAIN;
            let lr = self.eq_live_r.tick(lr as f64) * LIVE_GAIN;

            let l = (sl + mix * ll).tanh();
            let r = (sr + mix * lr).tanh();
            peak = peak.max(l.abs()).max(r.abs());

            // Write the final mix to the speakers AND cache it for the ring tee
            // so the analyzer sees the true blended output.
            self.last.push(trip_engine::out::Frame { l, r });
            write(i, l, r);
        }

        // Tee the blended (post-tanh) mix into the analyzer ring.
        let mut ring = self.ring.ring.lock().unwrap();
        for fr in self.last.iter() {
            ring.push_back((0.5 * (fr.l + fr.r)) as f32);
        }
        while ring.len() > RING_MAX {
            ring.pop_front();
        }
        drop(ring);

        if has_live {
            log_peak("tripengine: blended block peak", peak, n);
        }
    }
}

fn log_peak(tag: &str, peak: f64, _n: usize) {
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    if now_ms.saturating_sub(LAST.load(Ordering::Relaxed)) >= 3000 {
        log::info!("{tag} = {peak:.3}");
        LAST.store(now_ms, Ordering::Relaxed);
    }
}

fn audio_error(err: cpal::StreamError) {
    log::error!("tripengine audio error: {err}");
}

/// Gentle autoscrolling idle frame (matches the old `NullAnalyzer` shimmer).
/// Used before any samples arrive and during silence, so the feedback loop
/// always has a non-zero seed and the screen is never a dead black.
fn idle_frame(t: &Cell<f32>) -> AudioFrame {
    let secs = t.get();
    t.set(secs + 1.0 / 60.0);
    AudioFrame {
        bass: 0.5 + 0.5 * (secs * 0.7).sin() * 0.2,
        mid: 0.5 + 0.5 * (secs * 1.3 + 1.0).sin() * 0.2,
        treble: 0.5 + 0.5 * (secs * 2.1 + 2.0).sin() * 0.2,
        beat: 0.0,
        genre: Genre::Ambient,
        energy: 0.3,
        tempo: 0.2,
        sub_bass: 0.25,
        centroid: 0.4,
        crest: 0.2,
        flux: 0.05,
        rolloff: 0.3,
    }
}

/// Per-thread scratch buffers for [`analyze`]. The render thread calls this
/// once a frame, so we cache the Hann window, the FFT planner and the working
/// buffers here instead of allocating (and re-planning the FFT) every call.
struct AnalyzeScratch {
    window: Vec<f32>,
    hann: Vec<f32>,
    input: Vec<f32>,
    buffer: Vec<Complex<f32>>,
    planner: FftPlanner<f32>,
}

impl AnalyzeScratch {
    fn new() -> Self {
        let hann: Vec<f32> = (0..FFT_SIZE)
            .map(|i| {
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (FFT_SIZE - 1) as f32).cos()
            })
            .collect();
        Self {
            window: vec![0.0; FFT_SIZE],
            hann,
            input: vec![0.0; FFT_SIZE],
            buffer: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            planner: FftPlanner::new(),
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<Option<AnalyzeScratch>> = RefCell::new(None);
}

fn with_scratch<R>(f: impl FnOnce(&mut AnalyzeScratch) -> R) -> R {
    SCRATCH.with(|s| {
        let mut slot = s.borrow_mut();
        if slot.is_none() {
            *slot = Some(AnalyzeScratch::new());
        }
        f(slot.as_mut().expect("scratch initialized"))
    })
}

/// Analyze the ring: take the newest FFT_SIZE samples, apply a Hann window,
/// FFT, and derive band energies + a beat-onset pulse.
///
/// Bands are computed as mean FFT magnitude (normalized by window power),
/// converted to dB, then mapped 0..1 over [DB_FLOOR, DB_CEIL]. Attack/release
/// smoothing (fast attack, slow release) gives punchy but stable animation.
fn analyze(
    ring: &mut VecDeque<f32>,
    sample_rate: u32,
    beat_shared: &Mutex<BeatState>,
    prev: &AudioFrame,
) -> AudioFrame {
    with_scratch(|sc| {
        let window = &mut sc.window;
        let n = ring.len().min(FFT_SIZE);
        window.fill(0.0);
        for k in 0..n {
            window[FFT_SIZE - n + k] = ring[ring.len() - n + k];
        }

        // DC-block (subtract mean) + Hann window for clean low-frequency spread.
        let mean = window.iter().sum::<f32>() / window.len() as f32;
        let input = &mut sc.input;
        let hann = &sc.hann;
        for i in 0..FFT_SIZE {
            input[i] = (window[i] - mean) * hann[i];
        }

        let buffer = &mut sc.buffer;
        for (i, x) in input.iter().enumerate() {
            buffer[i].re = *x;
            buffer[i].im = 0.0;
        }
        let fft = sc.planner.plan_fft_forward(FFT_SIZE);
        fft.process(buffer);

        let nyquist = sample_rate as f32 * 0.5;
        let bin_hz = nyquist / (FFT_SIZE / 2) as f32;

        // Normalize magnitudes by the window power (sum of squares) so the result
        // is independent of FFT size / sample rate.
        let window_power: f32 = hann.iter().map(|w| w * w).sum::<f32>().max(1e-6);

        // Mean magnitude in a band (normalized by window power).
        let band_mag = |lo: f32, hi: f32| -> f32 {
            let lo_bin = ((lo / bin_hz) as usize).min(FFT_SIZE / 2);
            let hi_bin = ((hi / bin_hz) as usize).min(FFT_SIZE / 2 - 1);
            if hi_bin <= lo_bin {
                return 0.0;
            }
            let sum: f32 = (lo_bin..=hi_bin).map(|k| buffer[k].norm()).sum();
            sum / ((hi_bin - lo_bin + 1) as f32 * window_power)
        };
        let mags = (
            band_mag(20.0, 320.0),
            band_mag(180.0, 3800.0),
            band_mag(2200.0, 16000.0),
        );
        let sub_mag = band_mag(20.0, 80.0);

        // Calibration: reveal the raw band dB range once so DB_FLOOR/DB_CEIL can
        // be tuned against real signal levels.
        calibrate(mags);

        // magnitude dB -> 0..1.
        let map = |mag: f32| {
            let db = 20.0 * (mag + DB_EPS).log10();
            ((db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)).clamp(0.0, 1.0)
        };
        let raw = (map(mags.0), map(mags.1), map(mags.2));

        // --- Spectral shape descriptors -------------------------------------
        // Centroid/crest/rolloff are *shape* measures (ratios), so the raw
        // complex magnitudes work directly without the dB window normalization.
        let nyq_bin = FFT_SIZE / 2;
        let mut mag_sum = 0.0f32;
        let mut weighted = 0.0f32;
        let mut peak_mag = 0.0f32;
        for k in 1..=nyq_bin {
            let m = buffer[k].norm();
            mag_sum += m;
            weighted += m * k as f32;
            peak_mag = peak_mag.max(m);
        }
        let centroid = if mag_sum > 1e-9 {
            (weighted / mag_sum) / nyq_bin as f32
        } else {
            0.5
        };
        let mean_mag = mag_sum / nyq_bin as f32;
        let crest = if mean_mag > 1e-9 {
            ((peak_mag / mean_mag) - 1.0).min(9.0) / 9.0
        } else {
            0.0
        };
        let rolloff = if mag_sum > 1e-9 {
            let target = mag_sum * 0.85;
            let mut acc = 0.0f32;
            let mut rr = 0.5f32;
            for k in 1..=nyq_bin {
                acc += buffer[k].norm();
                if acc >= target {
                    rr = k as f32 / nyq_bin as f32;
                    break;
                }
            }
            rr
        } else {
            0.5
        };

        // Attack/release smoothing: rise fast on a transient, fall slowly — keeps
        // the motion punchy yet stable (no per-frame flicker). Shape measures get
        // gentler one-pole smoothing.
        let sm = |target: f32, v: f32| v + (if target > v { ATTACK } else { RELEASE }) * (target - v);
        let frame = (
            sm(raw.0, prev.bass),
            sm(raw.1, prev.mid),
            sm(raw.2, prev.treble),
        );
        let sub = sm(map(sub_mag), prev.sub_bass);
        let centroid = prev.centroid + (centroid - prev.centroid) * 0.18;
        let crest = prev.crest + (crest - prev.crest) * 0.30;
        let rolloff = prev.rolloff + (rolloff - prev.rolloff) * 0.15;
        // Spectral-flux proxy: frame-to-frame novelty of the smoothed bands,
        // scaled so a full swing approaches 1.
        let flux_raw =
            (frame.0 - prev.bass).abs() + (frame.1 - prev.mid).abs()
                + (frame.2 - prev.treble).abs();
        let flux = sm((flux_raw * 3.0).min(1.0), prev.flux);
        let frame = (frame.0, frame.1, frame.2);

        let beat = onset_pulse(beat_shared, frame.0, frame.1);

        // Energy: mean of the smoothed bands (0..1).
        let energy = ((frame.0 + frame.1 + frame.2) / 3.0).clamp(0.0, 1.0);

        // Tempo 0..1: 0 = sparse/slow interval, 1 = fast. interval is seconds
        // between beats; map ~2.0s (30bpm) -> 0 to ~0.25s (240bpm) -> 1.
        let b = beat_shared.lock().unwrap();
        let tempo = if b.interval > 0.0 && b.interval.is_finite() {
            ((2.0 - b.interval) / 1.75).clamp(0.0, 1.0)
        } else {
            prev.tempo
        };
        drop(b);

        // Genre heuristic from the spectral shape + tempo:
        //   - Loud, mid-forward, steady -> Melodic
        //   - Strong driving low end + fast tempo -> Rhythmic
        //   - Dominant bass, slower -> Bass
        //   - Quiet / sparse -> Ambient
        let genre = classify(frame.0, frame.1, frame.2, tempo, energy, prev.genre);

        // Smooth energy to avoid genre jitter.
        let energy = prev.energy + (energy - prev.energy) * 0.1;

        log_genre_change(prev.genre, genre);
        AudioFrame {
            bass: frame.0,
            mid: frame.1,
            treble: frame.2,
            beat,
            genre,
            energy,
            tempo,
            sub_bass: sub,
            centroid,
            crest,
            flux,
            rolloff,
        }
    })
}

/// Log once when the detected genre changes, so we can see the visuals track
/// the music's character at runtime.
fn log_genre_change(prev: Genre, cur: Genre) {
    static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
    if prev != cur {
        let tag = match cur {
            Genre::Ambient => 1,
            Genre::Bass => 2,
            Genre::Rhythmic => 3,
            Genre::Melodic => 4,
            Genre::Inactive => 5,
        };
        if LAST.swap(tag, std::sync::atomic::Ordering::Relaxed) != tag {
            log::info!("tripengine: genre -> {:?}", cur);
        }
    }
}

/// Pick a genre from the smoothed spectral shape + tempo. Uses a stable
/// "stay unless it clearly changes" rule so the visuals don't flicker between
/// genres frame to frame.
fn classify(bass: f32, mid: f32, treble: f32, tempo: f32, energy: f32, prev: Genre) -> Genre {
    if energy < 0.28 {
        return if prev == Genre::Ambient { Genre::Ambient } else { Genre::Ambient };
    }
    // Score each candidate from the current spectral profile.
    let melodic = mid + treble * 0.5;
    let rhythmic = bass * 0.6 + tempo * 0.7;
    let bassy = bass + tempo * 0.3;
    let mut g = if rhythmic >= melodic && rhythmic >= bassy {
        Genre::Rhythmic
    } else if melodic >= bassy {
        Genre::Melodic
    } else {
        Genre::Bass
    };
    // Hysteresis: don't bounce between neighbors unless tempo changed a lot.
    if prev == Genre::Bass && g == Genre::Rhythmic && tempo < 0.45 {
        g = Genre::Bass;
    }
    if prev == Genre::Rhythmic && g == Genre::Melodic && mid < treble {
        g = Genre::Rhythmic;
    }
    g
}

/// Emit the raw band dB once (guarded by a static) so we can calibrate the
/// DB_FLOOR/DB_CEIL window against real signal levels.
fn calibrate(mags: (f32, f32, f32)) {
    static DONE: AtomicBool = AtomicBool::new(false);
    if DONE.swap(true, Ordering::Relaxed) {
        return;
    }
    let to_db = |n: f32| 20.0 * (n + DB_EPS).log10();
    log::info!(
        "loopback: band dB (normalized) bass={:.1} mid={:.1} treble={:.1} (DB window [{DB_FLOOR},{DB_CEIL}])",
        to_db(mags.0),
        to_db(mags.1),
        to_db(mags.2),
    );
}

/// Beat onset: a rise of low/mid-band energy above a slowly-decaying baseline.
/// Uses both the bass and mid bands (percussive transients often live in the
/// mid, not just the sub) and runs a self-calibrating adaptive threshold so a
/// quiet track or a loud one both track without retuning constants. Times the
/// inter-beat interval used for the tempo feature (1/interval).
fn onset_pulse(beat_shared: &Mutex<BeatState>, bass: f32, mid: f32) -> f32 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f32())
        .unwrap_or(0.0);
    let mut b = beat_shared.lock().unwrap();

    // Percussive energy: bass-weighted with a mid kick-in. Weighted so a kick
    // (strong bass, some mid) and a snare/hi-hat (mid) both register.
    let energy = 0.7 * bass + 0.3 * mid;

    // Adaptive threshold: trigger when current energy is clearly above the
    // running baseline AND clears a small floor so dead air never fires.
    let trigger = energy > 0.10
        && energy > b.baseline * 1.22
        && energy > bass * 0.15
        && b.pulse < 0.4;

    let pulse = if trigger {
        (energy - b.baseline).min(1.0)
    } else {
        b.pulse * PULSE_DECAY
    };
    // On a fresh onset, push the pairwise interval into the rolling mean.
    // `last_beat` is assigned before the branch below, so the previous
    // "else if last_beat <= 0.0" reset was unreachable — folded away.
    if trigger {
        let gap = now - b.last_beat;
        b.prev_beat = b.last_beat;
        b.last_beat = now;
        if gap > 0.05 && gap < 2.5 {
            let w = 0.12;
            b.interval = if b.interval <= 0.0 { gap } else { b.interval * (1.0 - w) + gap * w };
        }
    }
    // Baseline adapts toward current energy (slowly), so the threshold tracks
    // the music's loudness envelope rather than staying fixed.
    b.baseline = b.baseline * 0.92 + energy * 0.08;
    b.pulse = pulse;
    pulse
}
