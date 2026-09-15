#version 450

// 5-Cell (Pentachoron) vertex shader: 4D→3D perspective projection with
// audio-driven 4D rotations and spring displacement.
//
// The raw 4D vertex position is rotated in XW / YW / ZW planes driven by
// the DSP audio bands, displaced outward by a bass-reactive "spring", then
// projected to 3D via perspective division along the W axis.
//
// Binding 0: DSPUniforms (UBO) — separate from the warp shader's UBO.

layout(location = 0) in vec4 in_position4D;

layout(binding = 0, std140) uniform DSPUniforms {
    mat4 mvp;
    vec4 audio_bands;        // x=sub-bass, y=bass, z=mid, w=treble
    vec4 rotation_angles;    // x=XW_angle, y=YW_angle, z=ZW_angle
    float spring_disp;
    float entropy_decay;
    float time;
} ubo;

layout(location = 0) out float v_entropy;
layout(location = 1) out float v_depth;

// --- 4D Rotation Matrix Generators ------------------------------------------
// Each builds a 4×4 matrix that rotates in one of the three hyperplanes
// involving the W axis.

mat4 rotateXW(float a) {
    float c = cos(a), s = sin(a);
    //              col0       col1     col2      col3
    return mat4(   c, 0, 0, s,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                  -s, 0, 0, c);
}

mat4 rotateYW(float a) {
    float c = cos(a), s = sin(a);
    return mat4(   1, 0, 0, 0,
                   0, c, 0,-s,
                   0, 0, 1, 0,
                   0, s, 0, c);
}

mat4 rotateZW(float a) {
    float c = cos(a), s = sin(a);
    return mat4(   1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, c, s,
                   0, 0,-s, c);
}

// ---------------------------------------------------------------------------

void main() {
    vec4 p = in_position4D;

    // Spring physics: push vertices outward from centre, scaled by bass.
    p.xyz += normalize(p.xyz) * (ubo.spring_disp * ubo.audio_bands.y);

    // Audio-driven 4D rotations (bass → XW, mid → YW, treble → ZW).
    p = rotateXW(ubo.rotation_angles.x) * p;
    p = rotateYW(ubo.rotation_angles.y) * p;
    p = rotateZW(ubo.rotation_angles.z) * p;

    // Perspective projection: 4D → 3D.
    float w_factor = 1.0 / (2.5 - p.w);
    vec3 pos3D = p.xyz * w_factor;

    v_entropy = ubo.entropy_decay;
    v_depth   = w_factor;
    gl_Position = ubo.mvp * vec4(pos3D, 1.0);
}
