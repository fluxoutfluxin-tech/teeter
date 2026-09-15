#version 450

// "Starburst" â€” a rotating spiky star with beat-pulsed arms.
//
// Same descriptor contract as warp.frag (set 0: UBO binding 0, texPrev
// binding 1, aspect in the uniform block), so it plugs into the existing
// single-pass pipeline with no renderer binding changes.
//
// Look: polar UV is bent into a star-shaped wave so rays radiate out of a
// bright core. The star slowly rotates; each arm ripples with the bands and
// the whole thing pulses outward on the bass/beat. Colors run a full hue
// spin then desaturate, and a Reinhard tone-map keeps the bright core from
// clipping into a flat white smudge.

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

    vec2 ap = (length(p) < 1e-5) ? vec2(1e-5, 0.0) : p;
    float ang = atan(ap.y, ap.x);
    float r = length(p);

    // Star field: radial wave creates N-pointed arms. Audio pushes the size.
    float points = 9.0;
    float star = sin(ang * points + u.iTime * 0.2) * 0.35 + 1.0;
    float pulse = 1.0 + 0.4 * u.bass + 0.3 * u.beat + 0.25 * u.zoom;
    r += star * 0.10 * pulse * (0.6 + 0.4 * u.mid);

    // Slow rotation of the whole burst.
    ang += u.rotate * 0.5 + u.presetSeed * 6.2831 * 0.2 + u.iTime * 0.18;

    vec2 rp = vec2(cos(ang), sin(ang)) * r;

    // Ripple along the arms, gated to the high band for shimmering tips.
    rp += normalize(rp + 0.0001) * sin(r * 21.0 + u.iTime * 2.2) * u.treble * 0.06;
    rp += vec2(u.warpX, u.warpY) * 0.12 * (0.5 + 0.5 * u.bass);
    rp.x /= u.aspect;

    vec2 tex = rp + 0.5;
    tex = 0.5 + (tex - 0.5) * 0.98;
    tex = clamp(tex, 0.001, 0.999);

    vec4 col = texture(texPrev, tex);

    float hue = u.palette + u.bass * 0.4 + u.presetSeed + u.iTime * 0.05;
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

    // Beat flash: washes the whole burst once per beat.
    shifted += u.beat * 0.20 * (0.5 + 0.5 * sin(vec3(0.0, 1.0, 2.0) + u.presetSeed * 10.0));

    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    // Bright hot core, stronger on the bass.
    shifted += exp(-r * 21.0 + 1.6) * (0.06 + 0.18 * u.bass) * vec3(1.0, 0.85, 0.6);

    // Filmic finish: preserves detail, tames the hot core (no flat white).
    float energy = clamp(0.84*u.bass + 0.24*u.mid + 0.24*u.treble + 0.66*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    shifted = teeter_crossfade(shifted, tex, u.dissolve);
    outColor = vec4(shifted, 1.0);
}
