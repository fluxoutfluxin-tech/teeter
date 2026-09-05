#version 450

// "Vortex gyre" — a tight rotational swirl that folds the plane back on itself.
//
// Same descriptor contract as warp.frag (set 0: UBO binding 0, texPrev
// binding 1, aspect in the uniform block), so it plugs into the existing
// single-pass pipeline with no renderer binding changes.
//
// Look: the angle is offset by a twist that is strongest near the center and
// falls off outward, so the field curls into a closed gyre (like a drain) and
// the tail wraps around the front. Bass and the rotate knob spin the gyre
// harder; mid folds wavy bands into it, and the beat pulses the point-eye at
// the center. Distinct from warp3's three-fold fold and warp1's sinusoidal
// wobble — this one is a single omnidirectional vortex.

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

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 p = uv - 0.5;
    p.x *= u.aspect;

    // Polar coordinates.
    float r = length(p);
    float a = atan(p.y, p.x);

    // Gyre twist: strongest at the center, easing off outward so the field
    // coils into a closed vortex. Spin hardens with bass and the rotate knob.
    float rot = u.rotate * 0.8 + u.presetSeed * 6.2831 * 0.30
              + u.iTime * (0.15 + 0.35 * u.bass);
    a += rot * (exp(-r * 2.5) * 3.0 + 0.40);

    // Mid folds soft wavy bands into the vortex (weaker on the outer rim).
    a += 0.50 * sin(r * 8.0 + u.iTime * 1.5) * u.mid * (1.0 - r * 0.5);
    // Warp stick sloshes the whole eye sideways as it swirls.
    a += u.warpX * 0.20 * sin(u.iTime * 0.5 + u.presetSeed * 8.0);

    // Slight inward contraction (feedback) + audio zoom toward center.
    float rr = r * (1.0 + 0.34 * u.zoom + 0.26 * u.bass) * 0.985;

    vec2 rp = vec2(cos(a), sin(a)) * rr;
    rp.x /= u.aspect; // keep sampling isotropic in UV space
    vec2 tex = rp + 0.5;
    tex = 0.5 + (tex - 0.5) * 0.985;

    vec4 col = texture(texPrev, tex);

    // Hue wash: full-color spin (with a little angle-based drift) while the
    // palette knob shifts it by hand; treble adds a faint neon edge.
    float hue = u.palette + u.bass * 0.45 + u.iTime * 0.07;
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
    shifted = mix(vec3(luma), shifted, 0.72);

    // Beat flash (soft) over the whole field.
    float flash = u.beat * 0.22;
    shifted += flash * (0.5 + 0.5 * sin(vec3(0.0, 1.0, 2.0) + u.presetSeed * 10.0));

    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    // Point-eye at the center: bright on the bass/beat, teal-tinted.
    float d = length(rp);
    shifted += exp(-d * 20.0 + 2.0) * (0.08 + 0.22 * u.bass + 0.10 * u.beat) * vec3(0.45, 0.75, 1.0);
    // Outer rim ring so the gyre reads as a closed loop on treble.
    shifted += u.treble * 0.04 * (0.5 + 0.5 * sin(d * 30.0 - u.iTime * 3.0)) * vec3(0.9, 0.8, 1.0);

    // Filmic finish: preserves detail, tames highlights (no flat white).
    float energy = clamp(0.84*u.bass + 0.24*u.mid + 0.24*u.treble + 0.66*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    outColor = vec4(shifted, 1.0);
}
