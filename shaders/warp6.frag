#version 450

// "Liquid" — domain-warped metaball blobs driven by mid/treble.
//
// Same descriptor contract as warp.frag (set 0: UBO binding 0, texPrev
// binding 1, aspect in the uniform block), so it plugs into the existing
// single-pass pipeline with no renderer binding changes.
//
// Look: a couple of octaves of cheap value noise are warped in on themselves
// (classic "domain warping") so the feedback surface shimmers like liquid.
// The blobs swell with the mid band and the sample distortion grows with the
// high band, giving a watery, organic slide instead of a hard geometric fold.
// Colors run a warm hue spin and tone-map so the surface keeps its shading.

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

float hash2(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.45);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    // Bilinear blend of 4 hashed corners.
    float a = hash2(i);
    float b = hash2(i + vec2(1.0, 0.0));
    float c = hash2(i + vec2(0.0, 1.0));
    float d = hash2(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

void main() {
    vec2 p = uv - 0.5;
    p.x *= u.aspect;

    // Slow base drift so the liquid never freezes.
    float t = u.iTime * 0.12;
    vec2 drift = vec2(0.3 * sin(t + u.presetSeed * 6.2831),
                      0.3 * cos(t * 0.9 + u.presetSeed * 9.0));

    // Wobble strong on the mid band; that's the "swell".
    float swell = 0.5 + 0.5 * u.mid;

    // Domain-warp the coordinate a couple of octaves.
    vec2 q = p * 2.5 + drift;
    vec2 r = vec2(
        noise(q + vec2(1.7, 9.2)),
        noise(q + vec2(8.3, 2.8))
    );
    vec2 f = vec2(noise(q + 2.0 * r + vec2(0.0, 0.7) * u.treble),
                  noise(q + 2.0 * r + vec2(0.4, 0.1) * u.treble));

    // Map the noise into a UV displacement. `1.0 - f` accents the blobs.
    vec2 rp = p + (0.35 + 0.45 * swell) * (f - 0.5);
    rp += vec2(u.warpX, u.warpY) * 0.12 * (0.5 + 0.5 * u.bass);
    // Slow rotation drives the whole pool around.
    float ang = u.rotate * 0.4 + u.iTime * 0.05 + u.presetSeed;
    float ca = cos(ang);
    float sa = sin(ang);
    rp = vec2(ca * rp.x - sa * rp.y, sa * rp.x + ca * rp.y);
    rp *= 1.0 + 0.25 * u.zoom + 0.20 * u.bass;
    rp.x /= u.aspect;

    vec2 tex = rp + 0.5;
    tex = 0.5 + (tex - 0.5) * 0.975;
    tex = clamp(tex, 0.001, 0.999);

    vec4 col = texture(texPrev, tex);

    // Warm, wet color wash (amber/rose) that cycles over time.
    float hue = u.palette + u.bass * 0.25 + u.presetSeed * 0.5 + u.iTime * 0.04;
    float h = hue * 6.2831;
    float c = cos(h);
    float sm = sin(h);
    mat3 hueRot = mat3(
        vec3(0.299 + 0.701*c + 0.168*sm, 0.587 - 0.587*c + 0.330*sm, 0.114 - 0.114*c - 0.497*sm),
        vec3(0.299 - 0.299*c - 0.328*sm, 0.587 + 0.413*c + 0.035*sm, 0.114 - 0.114*c + 0.292*sm),
        vec3(0.299 - 0.300*c + 1.250*sm, 0.587 - 0.588*c - 1.050*sm, 0.114 + 0.886*c - 0.203*sm)
    );
    vec3 shifted = hueRot * col.rgb;
    float luma = dot(shifted, vec3(0.299, 0.587, 0.114));
    shifted = mix(vec3(luma), shifted, 0.82);

    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    // Beat flash (very soft, wet).
    shifted += u.beat * 0.15 * (0.5 + 0.5 * sin(vec3(0.0, 1.0, 2.0) + u.presetSeed * 10.0));

    // Audio-reactive liquid core: re-injects light so the feedback loop never
    // decays to black, and swells with the bass so it follows the music.
    float energy = clamp(0.84*u.bass + 0.24*u.mid + 0.24*u.treble + 0.66*u.beat, 0.0, 1.0);
    shifted += exp(-length(rp) * 9.0) * (0.08 + 0.55 * energy) * vec3(0.35, 0.75, 1.0);

    // Filmic finish: preserves detail, tames highlights (no flat white).
    shifted = teeter_colorize(shifted, energy);
    outColor = vec4(shifted, 1.0);
}
