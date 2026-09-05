#version 450

// "Triangle octopus" — a 3-fold (triangular) tentacle feedback vortex.
//
// Same descriptor contract as warp.frag (set 0: UBO binding 0, texPrev
// binding 1, aspect in the uniform block), so it plugs into the existing
// single-pass pipeline with no renderer binding changes.
//
// Look: polar UV is folded into 3 symmetric arms (triangle symmetry); audio-
// driven tentacle waves ripple down each arm, feeding back into a wet,
// kelp-y vortex. Colors rotate over a full-strength hue spin, then get
// desaturated so it stays soft instead of neon.

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

    // Radial coords around the center. Guard the exact-center pixel: atan(0,0)
    // is NaN in GLSL and would poison the feedback loop, spreading a garbage
    // wave across the whole frame. If we're at the center, keep the angle
    // stable instead.
    vec2 ap = (length(p) < 1e-5) ? vec2(1e-5, 0.0) : p;
    float ang = atan(ap.y, ap.x);
    float r = length(p);

    // Fold the angle into 3-fold triangle symmetry (mirror-reflected).
    float folds = 3.0;
    float a = fract(ang / (6.2831 / folds));
    a = abs(a - 0.5) * 2.0;              // 0..1 triangle
    a *= 6.2831 / folds * 0.5;            // back to radians, one arm sector

    // Slow twist of the whole vortex.
    a += u.rotate * 0.6 + u.presetSeed * 6.2831 * 0.2 + u.iTime * 0.15;

    // Rebuild a 3-fold symmetric position.
    vec2 rp = vec2(cos(a), sin(a)) * r;

    // Zoom toward center (audio pushes outward like a pulsing octopus).
    rp *= 1.0 + 0.35 * u.zoom + 0.25 * u.bass;

    // Tentacle waves: ripple down the arms, gated by the bands and the pulse.
    float arm = sin(r * 30.0 - u.iTime * 3.0) * u.treble
              + sin(r * 13.0 + u.iTime * 2.0) * u.mid
              + sin(r * 7.0 + u.presetSeed * 20.0) * u.bass;
    rp += normalize(rp + 0.0001) * arm * 0.08;
    rp += vec2(u.warpX, u.warpY) * 0.12 * (0.5 + 0.5 * u.bass);
    rp.x /= u.aspect;

    // Feedback sample + inward contraction. Clamp UV to [0,1] so the sampler
    // never reads outside the texture (avoids the wrap/edge taking over and
    // smearing garbage into the loop).
    vec2 tex = rp + 0.5;
    tex = 0.5 + (tex - 0.5) * 0.98;
    tex = clamp(tex, 0.001, 0.999);

    vec4 col = texture(texPrev, tex);

    // Full-strength hue rotation for a clear, continuous color cycle.
    float hue = u.palette + u.bass * 0.35 + u.presetSeed + u.iTime * 0.05;
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

    // Beat flash (soft).
    float flash = u.beat * 0.18;
    shifted += flash * (0.5 + 0.5 * sin(vec3(0.0, 1.0, 2.0) + u.presetSeed * 10.0));

    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    // Radial "octopus core" glow driven by bass (kept subtle so the tentacle
    // detail near the arms isn't washed out by a bright halo).
    shifted += exp(-r * 16.0 + 1.2) * (0.05 + 0.14 * u.bass) * vec3(0.4, 0.6, 0.9);

    // Filmic finish: preserves detail, tames highlights (no flat white).
    float energy = clamp(0.84*u.bass + 0.24*u.mid + 0.24*u.treble + 0.66*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    outColor = vec4(shifted, 1.0);
}
