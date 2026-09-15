#version 450

// warp18 â€” star mandala (sacred geometry).
//
// Radial spokes folded into a multi-point star whose point count rides the
// spectral crest. Flux slowly spins the star, bass tears the center open, and
// the beat snaps the facets outward like a blooming flower.

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

    // Star points count on crest: band-spiky overtone-rich mixes get more points.
    float points = 5.0 + floor(u.crest * 6.0) * (1.0 + floor(u.cent * 2.0)) * 0.0
                 + floor(u.crest * 4.0);
    points = clamp(points, 5.0, 12.0);

    float ang = u.iTime * (0.05 + 0.30 * u.flux) + u.rotate * 0.6 + u.presetSeed * 6.2831;
    float ca = cos(ang), sa = sin(ang);
    p = mat2(ca, -sa, sa, ca) * p;
    p *= 1.0 + 0.5 * u.bass - 0.35 * u.zoom;

    // Fold the polar angle into a star cross-section.
    float seg = 6.2831 / points;
    float rad = length(p);
    float a = atan(p.y, p.x);
    a = a - seg * floor(a / seg + 0.5);

    // Star facet: the wall rises/cuts depth; beat pops it further out.
    float depth = cos(a * points) * 0.5 + 0.5;
    rad = rad / (0.55 + 0.45 * depth + 0.25 * u.beat * depth);

    vec2 s = rad * vec2(cos(a), sin(a));

    // Folded "mandala" rings; sub-bass breathes the fold radius.
    s *= 1.5 + 0.6 * sin(u.iTime * 1.3 + u.subBass * 4.0) * u.flux;
    s += 0.05 * (1.0 + u.treble) * vec2(sin(s.y * 26.0), cos(s.x * 26.0));

    vec2 tex = fract(vec2(s.x / u.aspect, s.y) + 0.5);
    vec3 col = texture(texPrev, tex).rgb;

    // Star walls light up on bass; radial spokes on treble.
    float w = clamp(1.0 - abs(a) / (seg * 0.5), 0.0, 1.0);
    col += w * w * (0.05 + 0.45 * u.bass) * vec3(1.0, 0.55, 0.35);
    col += 0.10 * u.treble * exp(-6.0 * abs(a)) * vec3(0.8, 0.9, 1.0);
    col += 0.03 * u.rolloff;

    float h = u.palette * 6.2831 + u.presetSeed * 3.0 + a * 0.7;
    float c = cos(h), sm = sin(h);
    mat3 hueRot = mat3(
        vec3(0.299 + 0.701*c + 0.168*sm, 0.587 - 0.587*c + 0.330*sm, 0.114 - 0.114*c - 0.497*sm),
        vec3(0.299 - 0.299*c - 0.328*sm, 0.587 + 0.413*c + 0.035*sm, 0.114 - 0.114*c + 0.292*sm),
        vec3(0.299 - 0.300*c + 1.250*sm, 0.587 - 0.588*c - 1.050*sm, 0.114 + 0.886*c - 0.203*sm)
    );
    vec3 shifted = hueRot * col.rgb;
    float luma = dot(shifted, vec3(0.299, 0.587, 0.114));
    shifted = mix(vec3(luma), shifted, 0.82);
    // Center bloom.
    shifted += exp(-rad * rad * 18.0) * (0.10 + 0.5 * u.bass) * vec3(1.0, 0.7, 0.5);
    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    float energy = clamp(0.85*u.bass + 0.45*u.mid + 0.55*u.treble + 0.60*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    shifted = teeter_crossfade(shifted, tex, u.dissolve);
    outColor = vec4(shifted, 1.0);
}