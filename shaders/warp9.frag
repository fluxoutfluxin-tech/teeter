#version 450

// "Spiral galaxy" — a rotating, winding galaxy of curved arms.
//
// Same descriptor contract as warp.frag (set 0: UBO binding 0, texPrev
// binding 1, aspect in the uniform block), so it plugs into the existing
// single-pass pipeline with no renderer binding changes.
//
// Look: the plane is mapped to polar coordinates and the angle is offset by a
// log-radius term, which curls the arms into a logarithmic spiral. The whole
// thing rotates slowly; mid/tetreble ripple along the arms and beat pulses the
// bright nucleus. Hue spins with a full-color wash and the star-bright core is
// tone-mapped so it never clips to flat white.

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

    // Polar coordinates (guard log against the exact center).
    float r = length(p) + 1e-4;
    float a = atan(p.y, p.x);

    // Spiral winding: angle grows with log radius so arms curl around the core.
    // A stable per-preset twist offsets the whole spiral; the bass pushes it.
    float twist = u.presetSeed * 6.2831 + u.rotate * 0.6
                + u.iTime * (0.10 + 0.20 * u.mid);
    float logr = log(r);
    a += twist * 2.0 * (1.0 + 0.30 * sin(logr * 5.0 + u.iTime));

    // Treble ripples radially along the arms; mid fans them out.
    a += 0.30 * sin(r * 30.0 - u.iTime * 2.0) * u.treble;
    a += 0.20 * u.mid * sin(a * 2.0 + u.iTime * 0.7);

    // Slight inward contraction (feedback) so arms shed toward the core.
    float rr = r * (1.0 + 0.36 * u.zoom + 0.22 * u.bass) * 0.985;

    vec2 rp = vec2(cos(a), sin(a)) * rr;
    rp.x /= u.aspect; // keep sampling isotropic in UV space
    vec2 tex = rp + 0.5;
    tex = 0.5 + (tex - 0.5) * 0.985;

    vec4 col = texture(texPrev, tex);

    // Hue wash: full-color spin, pulled toward warm on bass, always cycling.
    float hue = u.palette + u.bass * 0.45 + u.iTime * 0.05 + u.presetSeed * 0.5;
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

    // Beat flash (soft) over the whole galaxy.
    float flash = u.beat * 0.22;
    shifted += flash * (0.5 + 0.5 * sin(vec3(0.0, 1.0, 2.0) + u.presetSeed * 10.0));

    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    // Bright nucleus glows on the bass/beat; treble adds faint arm dust.
    float d = length(rp);
    shifted += exp(-d * 24.0 + 2.0) * (0.10 + 0.26 * u.bass + 0.10 * u.beat) * vec3(1.0, 0.85, 0.7);
    shifted += 0.03 * u.treble * vec3(1.0, 1.0, 0.95);

    // Filmic finish: preserves detail, tames highlights (no flat white).
    float energy = clamp(0.84*u.bass + 0.24*u.mid + 0.24*u.treble + 0.66*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    outColor = vec4(shifted, 1.0);
}
