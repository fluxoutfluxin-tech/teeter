#version 450

// warp14 â€” kaleido mandala (sacred geometry).
//
// Folds the feedback frame into a mirrored pie of N petals like a kaleidoscope,
// with radial "heartbeat" rings riding sub-bass, a centroid-driven petal count
// and a hue wash keyed to the petal angle. Classic dissolve tail on top.

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
    float dissolve;       // warp-switch crossfade weight 0..1
    float palette;        // palette shift 0..1
    float presetSeed;     // stable random per preset
    float aspect;         // framebuffer width / height
    float subBass;        // sub-60Hz energy     0..1
    float cent;       // spectral centroid   0..1
    float crest;          // spectral crestfactor 0..1
    float flux;           // spectral flux (motion) 0..1
    float rolloff;        // high-end rolloff brightness 0..1
} u;

layout(set = 0, binding = 1) uniform sampler2D texPrev;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 p = uv - 0.5;
    p.x *= u.aspect;

    // Slow global spin, sped up by spectral flux.
    float ang = u.iTime * (0.08 + 0.20 * u.flux) + u.rotate * 0.8 + u.presetSeed * 6.2831;
    float ca = cos(ang), sa = sin(ang);
    p = mat2(ca, -sa, sa, ca) * p;
    p *= 1.0 + 0.30 * u.zoom + 0.35 * u.bass;

    // Fold into a mirrored pie-slice: the mandala petals. Petal count grows
    // with the spectral centroid (brighter mix -> sharper, denser mandala).
    float folds = 6.0 + floor(u.cent * 4.0);
    float seg = 6.2831 / folds;
    float rad = length(p);
    float a = atan(p.y, p.x);
    a = mod(a, seg);
    a = abs(a - seg * 0.5);
    vec2 m = rad * vec2(cos(a), sin(a));

    // Push into the mirror; the heartbeat rings pulse on sub-bass.
    m *= 1.7 + 0.45 * u.bass - 0.35 * u.zoom;
    float r = length(m);
    float ring = 0.5 + 0.5 * sin(r * 52.0 - u.iTime * 4.2 + u.subBass * 6.0);
    m += 0.06 * ring * normalize(m + 1e-4) * (0.5 + 1.2 * u.treble);
    // Beat shatter along the spokes.
    m += 0.045 * (0.4 + u.beat) * vec2(sin(m.y * 90.0), cos(m.x * 90.0));

    vec2 tex = vec2(m.x / u.aspect, m.y) + 0.5;
    vec3 col = texture(texPrev, fract(tex)).rgb;

    // Hue wash keyed to the petal angle + palette drift.
    float h = a / seg * 6.2831 + u.palette * 6.2831 + u.presetSeed * 2.0;
    float c = cos(h), sm = sin(h);
    mat3 hueRot = mat3(
        vec3(0.299 + 0.701*c + 0.168*sm, 0.587 - 0.587*c + 0.330*sm, 0.114 - 0.114*c - 0.497*sm),
        vec3(0.299 - 0.299*c - 0.328*sm, 0.587 + 0.413*c + 0.035*sm, 0.114 - 0.114*c + 0.292*sm),
        vec3(0.299 - 0.300*c + 1.250*sm, 0.587 - 0.588*c - 1.050*sm, 0.114 + 0.886*c - 0.203*sm)
    );
    vec3 shifted = hueRot * col.rgb;

    // Soft but vivid.
    float luma = dot(shifted, vec3(0.299, 0.587, 0.114));
    shifted = mix(vec3(luma), shifted, 0.80);

    // Center bloom (bass-lifted).
    shifted += exp(-rad * rad * 14.0) * (0.12 + 0.55 * u.bass) * vec3(0.6, 0.8, 1.0);
    // Baseline so the feedback never fully drains.
    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    float energy = clamp(0.80*u.bass + 0.45*u.mid + 0.55*u.treble + 0.60*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    shifted = teeter_crossfade(shifted, tex, u.dissolve);
    outColor = vec4(shifted, 1.0);
}