#version 450

// The MilkDrop-style feedback "warp" fragment shader.
//
// Samples the PREVIOUS frame stored in the `feedback` texture (texPrev) and
// applies an audio-reactive UV warp + rotation + hue wash, writing the result
// to the swapchain. Afterwards the CPU-side renderer blits the frame back into
// the feedback texture, so every frame wraps the previous one.
//
// Descriptor bindings (matches the Rust renderer):
//   set 0, binding 0 -> FrameUniforms (UBO)
//   set 0, binding 1 -> texPrev (feedback sampler)

#include "teeter_common.h"

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform FrameUniforms {
    float iTime;          // seconds since start
    float bass;           // low band energy   0..1
    float mid;            // mid band energy   0..1
    float treble;         // high band energy  0..1
    float beat;           // beat pulse        0..1
    float zoom;           // right-stick zoom  -1..1
    float warpX;          // left-stick warpX  -1..1
    float warpY;          // left-stick warpY  -1..1
    float rotate;         // touch / stick rotation -1..1
    float dissolve;       // triggers dissolve
    float palette;        // palette shift 0..1
    float presetSeed;     // stable random per preset
    float aspect;         // framebuffer width / height
} u;

layout(set = 0, binding = 1) uniform sampler2D texPrev;

// tiny hash for stable per-preset noise
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 p = uv - 0.5;

    // Aspect-correct: scale x so the coordinate space is isotropic. This keeps
    // circles circular (and the rotation/zoom/warp symmetric) on any
    // window/framebuffer ratio instead of stretching into ellipses.
    p.x *= u.aspect;

    // Base rotation from input + auto drift.
    float ang = u.rotate * 0.6 + u.presetSeed * 6.2831 * 0.2
              + 0.5 * sin(u.iTime * 0.13 + u.presetSeed * 40.0);
    float ca = cos(ang);
    float sa = sin(ang);
    vec2 rp = vec2(ca * p.x - sa * p.y, sa * p.x + ca * p.y);

    // Zoom toward center.
    rp *= 1.0 + 0.35 * u.zoom + 0.25 * u.bass;

    // Audio-reactive warp (the "teeter" wobble): displace by band energies.
    vec2 warp = vec2(
        0.06 * sin(rp.y * 18.0 + u.iTime * 2.2) * u.treble,
        0.06 * sin(rp.x * 14.0 - u.iTime * 1.7) * u.mid
    );
    warp += vec2(u.warpX, u.warpY) * 0.12 * (0.5 + 0.5 * u.bass);
    warp.x /= u.aspect; // keep the displaced sampling isotropic in UV space
    rp += warp;

    // Sample the previous frame's texture (feedback source).
    vec2 tex = rp + 0.5;
    // Slight inward sampling pushes color toward center over time (feedback
    // contraction), a classic MilkDrop trait.
    tex = 0.5 + (tex - 0.5) * 0.985;

    vec4 col = texture(texPrev, tex);

    // Color wash: shift hue with palette drift, bass, and continuous cycling.
    // The `+ iTime * 0.05` term rotates the whole color wheel over time
    // (full cycle ~20s) so the palette never sits still.
    float hue = u.palette + u.bass * 0.4 + u.presetSeed + u.iTime * 0.05;
    float h = hue * 6.2831;
    // Full-strength hue rotation so the cycle is clearly visible...
    float s = 1.0;
    float c = s * cos(h);
    float sm = s * sin(h);
    // Standard hue-rotate matrix (applied to RGB).
    mat3 hueRot = mat3(
        vec3(0.299 + 0.701*c + 0.168*sm, 0.587 - 0.587*c + 0.330*sm, 0.114 - 0.114*c - 0.497*sm),
        vec3(0.299 - 0.299*c - 0.328*sm, 0.587 + 0.413*c + 0.035*sm, 0.114 - 0.114*c + 0.292*sm),
        vec3(0.299 - 0.300*c + 1.250*sm, 0.587 - 0.588*c - 1.050*sm, 0.114 + 0.886*c - 0.203*sm)
    );
    vec3 shifted = hueRot * col.rgb;

    // ...then desaturate afterwards so the full-rotation colors stay soft and
    // easy on the eyes instead of turning neon.
    float luma = dot(shifted, vec3(0.299, 0.587, 0.114));
    shifted = mix(vec3(luma), shifted, 0.72);

    // Beat flash (softer).
    float flash = u.beat * 0.22;
    shifted += flash * (0.5 + 0.5 * sin(vec3(0.0, 1.0, 2.0) + u.presetSeed * 10.0));

    // Subtle baseline brightness so it never fully vanishes (milkdrop decay).
    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    // Add a faint radial pulsing "eye" driven by bass (toned way down).
    float d = length(rp);
    shifted += exp(-d * 22.0 + 2.0) * (0.06 + 0.16 * u.bass) * vec3(0.4, 0.7, 1.0);

    // Filmic finish: preserves detail, tames highlights (no flat white).
    float energy = clamp(0.84*u.bass + 0.24*u.mid + 0.24*u.treble + 0.66*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    shifted = teeter_crossfade(shifted, tex, u.dissolve);
    outColor = vec4(shifted, 1.0);
}
