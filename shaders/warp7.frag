#version 450

// "Confetti embers" â€” scattered spinning sprites that spark on the beat.
//
// Same descriptor contract as warp.frag (set 0: UBO binding 0, texPrev
// binding 1, aspect in the uniform block), so it plugs into the existing
// single-pass pipeline with no renderer binding changes.
//
// Look: the UV is quantized into a coarse grid of "cells"; each cell gets a
// per-preset hash that offsets a small sprite. On a beat the sprites scatter
// (and the feedback streaks them), so embers fly out of the center and settle
// back between beats â€” like a machine-lights confetti burst that ghosts
// through the feedback loop.

#include "teeter_common.h"

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform FrameUniforms {
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
    float aspect;
} u;

layout(set = 0, binding = 1) uniform sampler2D texPrev;

float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 p = uv - 0.5;
    p.x *= u.aspect;

    // Cell grid. `2.0 +` keeps a couple of active rounds of confetti.
    const float CELLS = 9.0;
    vec2 cell = floor(p * CELLS);
    vec2 local = fract(p * CELLS) - 0.5;

    // Per-cell deterministic hashes (stable per preset).
    float h = hash21(cell * 0.31 + u.presetSeed);
    float h2 = hash21(cell * 1.7 + u.presetSeed + 3.1);

    // Sprite offset within the cell + a spin that turns with time.
    vec2 offset = (vec2(h, h2) - 0.5) * 0.9;
    float spin = (h - 0.5) * 2.0 * (u.iTime * 0.3 + u.presetSeed);
    float ca = cos(spin);
    float sa = sin(spin);

    // Confetti scatters outward on the beat (feedback streaks it).
    vec2 rp = cell / CELLS + local;
    float burst = u.beat;
    rp += offset * (0.6 + 1.6 * burst);
    rp = vec2(ca * rp.x - sa * rp.y, sa * rp.x + ca * rp.y);

    // Embers swirl toward center with the bass; beat throws them outward.
    rp *= 0.7 + 0.30 * u.zoom + 0.5 * u.bass - 0.5 * burst;
    rp += vec2(u.warpX, u.warpY) * 0.10 * (0.5 + 0.5 * u.bass);
    rp.x /= u.aspect;

    vec2 tex = rp + 0.5;
    tex = 0.5 + (tex - 0.5) * 0.98;
    tex = clamp(tex, 0.001, 0.999);

    vec4 col = texture(texPrev, tex);

    // Confetti is multicolored: tint per cell by its hash.
    float hue = u.palette + h + u.iTime * 0.03;
    float hh = hue * 6.2831;
    float c = cos(hh);
    float sm = sin(hh);
    mat3 hueRot = mat3(
        vec3(0.299 + 0.701*c + 0.168*sm, 0.587 - 0.587*c + 0.330*sm, 0.114 - 0.114*c - 0.497*sm),
        vec3(0.299 - 0.299*c - 0.328*sm, 0.587 + 0.413*c + 0.035*sm, 0.114 - 0.114*c + 0.292*sm),
        vec3(0.299 - 0.300*c + 1.250*sm, 0.587 - 0.588*c - 1.050*sm, 0.114 + 0.886*c - 0.203*sm)
    );
    vec3 shifted = hueRot * col.rgb;
    float luma = dot(shifted, vec3(0.299, 0.587, 0.114));
    shifted = mix(vec3(luma), shifted, 0.82);

    // Beat spark: brightening every cell at once.
    shifted += u.beat * 0.18 * (0.5 + 0.5 * sin(vec3(0.0, 1.0, 2.0) + h * 6.2831));

    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    // Audio-reactive ember glow: re-injects light so the feedback never decays
    // to black, and brightens with the beat so the burst follows the music.
    float energy = clamp(0.84*u.bass + 0.24*u.mid + 0.24*u.treble + 0.66*u.beat, 0.0, 1.0);
    shifted += exp(-length(rp) * 10.0) * (0.06 + 0.45 * energy) * vec3(1.0, 0.45, 0.7);

    // Filmic finish: preserves detail, tames highlights (no flat white).
    shifted = teeter_colorize(shifted, energy);
    shifted = teeter_crossfade(shifted, tex, u.dissolve);
    outColor = vec4(shifted, 1.0);
}
