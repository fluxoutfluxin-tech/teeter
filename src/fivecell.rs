//! 5-Cell (Pentachoron) geometry and DSP UBO for Vulkan overlay rendering.
//!
//! A regular 5-cell (4-simplex, 5 vertices / 10 edges) rendered as additive
//! wireframe lines over the feedback warp background.  The 4D→3D projection
//! and audio-driven rotations live in the vertex shader; this module supplies
//! the vertex/index data and the Rust-side UBO layout.

/// Pentagon vertices in 4D space (regular pentachoron centered at the origin).
pub fn vertices() -> [[f32; 4]; 5] {
    let inv_sqrt5 = 1.0 / 5.0_f32.sqrt();
    let sqrt5 = 5.0_f32.sqrt();
    [
        [ 1.0,  1.0,  1.0, -inv_sqrt5],
        [-1.0, -1.0,  1.0, -inv_sqrt5],
        [-1.0,  1.0, -1.0, -inv_sqrt5],
        [ 1.0, -1.0, -1.0, -inv_sqrt5],
        [ 0.0,  0.0,  0.0,  sqrt5 - inv_sqrt5],
    ]
}

/// All 10 edges of the pentachoron (20 u16 indices, one pair per edge).
pub fn edge_indices() -> [u16; 20] {
    [0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 4]
}

/// std140-aligned UBO matching the GLSL `DSPUniforms` block.
///
/// Field layout (total 112 bytes = 14 × f32):
/// ```text
///  0..63  mat4  mvp                (column-major)
/// 64..79  vec4  audio_bands       (sub-bass, bass, mid, treble)
/// 80..95  vec4  rotation_angles   (XW, YW, ZW, unused)
/// 96      float spring_disp
///100      float entropy_decay
///104      float time
///108      float (padding / reserved)
/// ```
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DSPUniforms {
    pub mvp: [f32; 16],
    pub audio_bands: [f32; 4],
    pub rotation_angles: [f32; 4],
    pub spring_disp: f32,
    pub entropy_decay: f32,
    pub time: f32,
    pub _padding: f32,
}

/// Byte size of the UBO (must match `sizeof(DSPUniforms)` in the shader).
pub const UNIFORM_SIZE: usize = std::mem::size_of::<DSPUniforms>();

impl DSPUniforms {
    /// Build the per-frame UBO from the current audio snapshot.
    pub fn from_audio(
        mvp: glam::Mat4,
        audio: &crate::audio::AudioFrame,
        time: f32,
    ) -> Self {
        let angles = [
            time * (0.3 + audio.sub_bass * 1.5), // XW driven by sub-bass
            time * (0.2 + audio.mid * 1.2),      // YW driven by mid
            time * (0.15 + audio.treble * 1.0),   // ZW driven by treble
            0.0,
        ];
        Self {
            mvp: mvp.to_cols_array(),
            audio_bands: [audio.sub_bass, audio.bass, audio.mid, audio.treble],
            rotation_angles: angles,
            spring_disp: 0.15,
            entropy_decay: 0.30 + audio.energy * 0.70,
            time,
            _padding: 0.0,
        }
    }
}
