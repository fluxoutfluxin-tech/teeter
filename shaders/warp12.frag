#version 450

// "Surf / planar waves" — a flat X/Y shearing ocean rather than a polar fold.
//
// Same descriptor contract as warp.frag (set 0 binding 0 = UBO, binding 1 =
// texPrev), so it plugs into the existing pipeline with no renderer changes.
//
// Look: the UV is sheared by horizontal and vertical travelling waves (like an
// old audio-analyzer or a rippling water surface). Treble ruffles fine crests,
// mid drives bigger rolling swells, and bass lifts the contrast so the troughs
// go deep. The whole surface slopes slightly on the rotate knob. Hue stays
// closer to a fixed cool gradient that warms with bass.

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

void main() {
    vec2 p = uv - 0.5;
    p.x *= u.aspect;

    // Slow global drift + a soft shear on rotate.
    p = vec2(
        p.x * (1.0 + 0.08 * u.rotate),
        p.y + 0.10 * u.rotate * p.x
    );

    // Zoom toward center (bass breath).
    p *= 1.0 + 0.30 * u.zoom + 0.26 * u.bass;

    // Planar domain warp: big mid-driven swells + fine treble crests, doubling
    // into a shallow Lissajous cross so the surface rolls 2-dimensionally.
    vec2 warp = vec2(
        (0.10 * sin(p.y * 8.0 + u.iTime * 1.3) + 0.04 * sin(p.y * 34.0 - u.iTime * 2.6)) * (0.5 + u.mid),
        (0.10 * sin(p.x * 10.0 - u.iTime * 1.1) + 0.04 * cos(p.x * 29.0 + u.iTime * 2.9)) * (0.5 + u.mid)
    );
    warp += vec2(
        0.03 * sin(p.y * 90.0 + u.iTime * 3.0) * u.treble,
        0.03 * cos(p.x * 80.0 - u.iTime * 3.4) * u.treble
    );
    warp += vec2(u.warpX, u.warpY) * 0.10;
    warp.x /= u.aspect;
    p += warp;

    // Sample previous frame + inward contraction.
    vec2 tex = p + 0.5;
    tex = 0.5 + (tex - 0.5) * 0.98;

    vec4 col = texture(texPrev, tex);

    // Cool-to-warm gradient that warms with bass; gentle cycling.
    float hue = 0.55 + u.palette + u.bass * 0.35 + u.presetSeed * 0.5 + u.iTime * 0.02;
    float h = hue * 6.2831;
    float s = 1.0;
    float c = s * cos(h);
    float sm = s * sin(h);
    mat3 hueRot = mat3(
        vec3(0.299 + 0.701*c + 0.168*sm, 0.587 - 0.587*c + 0.330*sm, 0.114 - 0.114*c - 0.497*sm),
        vec3(0.299 - 0.299*c - 0.328*sm, 0.587 + 0.413*c + 0.035*sm, 0.114 - 0.114*c + 0.292*sm),
        vec3(0.299 - 0.300*c + 1.250*sm, 0.587 - 0.588*c - 1.050*sm, 0.114 + 0.886*c - 0.203*sm)
    );
    vec3 shifted = hueRot * col.rgb;
    float luma = dot(shifted, vec3(0.299, 0.587, 0.114));
    shifted = mix(vec3(luma), shifted, 0.78);

    // Beat flash.
    shifted += u.beat * 0.20 * (0.5 + 0.5 * sin(vec3(0.0, 1.0, 2.0) + u.presetSeed * 10.0));

    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    // Filmic finish: preserves detail, tames highlights (no flat white).
    float energy = clamp(0.84*u.bass + 0.24*u.mid + 0.24*u.treble + 0.66*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    outColor = vec4(shifted, 1.0);
}
