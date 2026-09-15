#version 450

// 5-Cell (Pentachoron) fragment shader.
//
// Renders the wireframe edges as glowing lines with a cool→warm colour ramp
// driven by the vertex-projected depth and the entropy-decay intensity from
// the DSP UBO.  Designed to be blended ADDITIVELY over the feedback warp
// background so the geometry reads as a luminous wireframe overlay.

layout(location = 0) in float v_entropy;
layout(location = 1) in float v_depth;

layout(location = 0) out vec4 outColor;

void main() {
    // Intensity: entropy decay × depth factor, clamped to a visible range.
    float intensity = clamp(v_entropy * v_depth * 1.2, 0.0, 1.0);

    // Colour ramp: cool blue (near) → warm magenta (far).
    vec3 cool = vec3(0.25, 0.55, 1.0);
    vec3 warm = vec3(1.0, 0.35, 0.55);
    vec3 col  = mix(warm, cool, v_depth);

    // Output with alpha for additive blending (alpha ignored in hardware
    // additive blend mode, but kept for pipeline compatibility).
    outColor = vec4(col * intensity, intensity * 0.6);
}
