#version 450

// warp20 â€” lowpoly shards (chromatic).
//
// The feedback plane shatters into flat polygon shards (hash grid -> centroid
// samples). Shard count follows the spectral centroid, facet shading follows
// crest, and each shard glints with a chromatic offset as the beat lands.
// This is the "grand finale" warp: heavy, crystalline, definitely not MilkDrop.

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

    // Broad rotation + zoom.
    float ang = u.rotate * 0.7 + u.presetSeed * 6.2831 + 0.4 * sin(u.iTime * 0.16);
    float ca = cos(ang), sa = sin(ang);
    p = mat2(ca, -sa, sa, ca) * p;
    p *= 1.0 + 0.4 * u.zoom + 0.35 * u.bass;

    // Shard grid: centroid picks the shard density (few huge chunks <-> many facets).
    float g = 16.0 + 44.0 * u.cent;
    vec2 cellP = p * g;
    vec2 cell = round(cellP);
    vec2 fp = cellP - cell;
    vec2 seed = cell;

    // Per-shard wobble on flux keeps everything alive.
    float w = hash(seed);
    cellP += 0.10 * (1.0 + u.flux) * vec2(cos(w * 40.0 + u.iTime * 0.5), sin(w * 90.0 + u.iTime * 0.3)) * (1.0 - length(fp));

    // Sample from the shard center so corners splay into flat facets.
    vec2 t0 = (cellP - fp * 0.0) / g;
    vec2 tex = fract(vec2(t0.x / u.aspect, t0.y) + 0.5);

    // Chromatic aberration offset by crest, plus a bass lens push.
    vec2 chroma = 0.012 * u.crest * vec2(1.0, 0.0) + 0.03 * u.bass * vec2(u.warpX, u.warpY);
    chroma.x /= u.aspect;
    vec3 col;
    col.r = texture(texPrev, tex + chroma).r;
    col.g = texture(texPrev, tex).g;
    col.b = texture(texPrev, tex - chroma).b;

    // Facet shading: shard center bright, edges darker (flat look).
    float facet = 1.0 - 0.55 * length(fp);
    facet = mix(facet, clamp(facet + 0.3 * u.beat, 0.0, 1.2), 0.5);
    col *= facet * (0.45 + 0.45 * u.subBass);
    // Edge glint line on the shard boundaries.
    float edgeGlow = 1.0 - smoothstep(0.02, 0.06, abs(length(fp) - 0.5));
    col += edgeGlow * (0.05 + 0.4 * u.mid) * vec3(0.5, 0.7, 1.0);
    col += 0.04 * u.rolloff;

    float h = u.palette * 6.2831 + w * 6.2831;
    float c = cos(h), sm = sin(h);
    mat3 hueRot = mat3(
        vec3(0.299 + 0.701*c + 0.168*sm, 0.587 - 0.587*c + 0.330*sm, 0.114 - 0.114*c - 0.497*sm),
        vec3(0.299 - 0.299*c - 0.328*sm, 0.587 + 0.413*c + 0.035*sm, 0.114 - 0.114*c + 0.292*sm),
        vec3(0.299 - 0.300*c + 1.250*sm, 0.587 - 0.588*c - 1.050*sm, 0.114 + 0.886*c - 0.203*sm)
    );
    vec3 shifted = hueRot * col.rgb;
    float luma = dot(shifted, vec3(0.299, 0.587, 0.114));
    shifted = mix(vec3(luma), shifted, 0.78);
    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    float energy = clamp(0.85*u.bass + 0.45*u.mid + 0.55*u.treble + 0.60*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    shifted = teeter_crossfade(shifted, tex, u.dissolve);
    outColor = vec4(shifted, 1.0);
}