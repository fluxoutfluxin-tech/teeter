#version 450

// "Mirror bloom" â€” a 4-fold kaleidoscope of mirrored wedges.
//
// Same descriptor contract as warp.frag (set 0: UBO binding 0, texPrev
// binding 1, aspect in the uniform block), so it plugs into the existing
// single-pass pipeline with no renderer binding changes.
//
// Look: the plane is folded into 4 mirrored radial sectors (kaleidoscope),
// so any motion is reflected symmetrically into a flower-like bloom. A slow
// drift rotates the whole flower; bass opens it outward and beat adds a
// soft cross-shaped flash. Colors run a full hue spin, then get desaturated
// and tone-mapped so brights keep detail.

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

    // Polar coords; guard center to avoid atan(0,0)=NaN poisoning feedback.
    vec2 ap = (length(p) < 1e-5) ? vec2(1e-5, 0.0) : p;
    float ang = atan(ap.y, ap.x);
    float r = length(p);

    // Merge the two mirrored halves of each sector so a wobble becomes a bloom
    // (triangle reflect). 4-fold mirrored kaleidoscope.
    float folds = 4.0;
    float a = fract(ang / (6.2831 / folds));
    a = abs(a - 0.5) * 2.0;
    a *= 6.2831 / folds * 0.5;

    // Slow flower drift from time + input rotation.
    a += u.rotate * 0.5 + u.presetSeed * 6.2831 * 0.25 + u.iTime * 0.12;

    vec2 rp = vec2(cos(a), sin(a)) * r;
    rp *= 1.0 + 0.32 * u.zoom + 0.24 * u.bass;

    // Radial ripple rings, gated by the bands.
    float ring = sin(r * 26.0 - u.iTime * 2.4) * u.treble
               + sin(r * 11.0 + u.iTime * 1.6) * u.mid
               + sin(r * 6.0 + u.presetSeed * 18.0) * u.bass;
    rp += normalize(rp + 0.0001) * ring * 0.07;
    rp += vec2(u.warpX, u.warpY) * 0.11 * (0.5 + 0.5 * u.bass);
    rp.x /= u.aspect;

    vec2 tex = rp + 0.5;
    tex = 0.5 + (tex - 0.5) * 0.98;
    tex = clamp(tex, 0.001, 0.999);

    vec4 col = texture(texPrev, tex);

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

    // Cross-flash on beat: two perpendicular soft bars through the center.
    float crossR = cos(a * folds * 0.5);
    float flash = u.beat * 0.20 * crossR * crossR;
    shifted += flash * (0.5 + 0.5 * sin(vec3(0.0, 1.0, 2.0) + u.presetSeed * 10.0));

    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    // Gentle rose glow at the center.
    shifted += exp(-r * 17.0 + 1.3) * (0.04 + 0.12 * u.bass) * vec3(0.8, 0.4, 0.7);

    // Filmic finish: preserves detail, tames highlights (no flat white).
    float energy = clamp(0.84*u.bass + 0.24*u.mid + 0.24*u.treble + 0.66*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    shifted = teeter_crossfade(shifted, tex, u.dissolve);
    outColor = vec4(shifted, 1.0);
}
