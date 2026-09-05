#version 450

// Feedback target fragment shader.
//
// This runs when rendering INTO the feedback texture. It samples the current
// warped result (texCurrent) and applies gentle color grading/normalization
// so the loop neither explodes to white nor dies to black. The updated frame
// is what the next warp.frag pass reads as texPrev.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D texCurrent;

layout(set = 0, binding = 2) uniform FrameUniforms {
    float iTime;
    float bass;
    float mid;
    float treble;
    float beat;
    float zoom;
    float warpX;
    float warpY;
    float rotate;
    float dissolve;
    float palette;
    float presetSeed;
} u;

void main() {
    vec4 col = texture(texCurrent, uv);

    // Soft clip to keep the feedback loop stable.
    vec3 c = col.rgb / (1.0 + col.rgb);

    // Slight saturation boost.
    float luma = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(luma), c, 1.25);

    // Fade darkness near edges slightly to avoid corner burn-in.
    vec2 d = abs(uv - 0.5) * 1.4;
    float vignette = 1.0 - 0.35 * dot(d, d);
    c *= vignette;

    outColor = vec4(c, 1.0);
}
