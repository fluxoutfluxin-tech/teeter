#version 450

// warp15 â€” hex lattice (geometric).
//
// Samples the feedback frame through a honeycomb of hex cells whose size is
// driven by the spectral centroid, and lights the cell walls with mid + treble
// energy. High rolloff brightens the pattern so dark mixes still read.

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

void main() {
    vec2 p = uv - 0.5;
    p.x *= u.aspect;

    // Cell scale: centroid thins the lattice, bass grows it.
    float cell = 0.10 + 0.30 * u.bass - 0.12 * u.cent;
    cell = max(cell, 0.02);

    // Slow rotation + zoom pull.
    float ang = u.iTime * 0.03 + u.rotate * 0.4 + u.presetSeed * 6.2831;
    float ca = cos(ang), sa = sin(ang);
    p = mat2(ca, -sa, sa, ca) * p;
    p /= cell;
    p *= 1.0 + 0.35 * u.zoom;

    // Axial hex coords.
    vec2 ax = vec2(p.x - p.y * 0.5, p.y * 0.866025);

    // Cell + intra-cell position.
    vec2 ci = vec2(round(ax.x), round(ax.y));
    float dd = dot(vec2(p.x - ci.x, p.y - ci.y), vec2(p.x - ci.x, p.y - ci.y));

    // Honeycomb wall glow + beat pulse on the cell centers.
    vec2 cc = vec2(ci.x + 0.5 * ci.y, ci.y * 0.866025); // axial->cart approx
    vec2 cfull = vec2(cc.x - cc.y * 0.5, cc.y);

    vec2 tex = fract(uv + 0.10 * (1.0 + u.flux) * sin(6.2831 * (cfull * 0.5 + u.iTime * 0.1)));
    // Displace within-cell content toward the cell center for a faceted look,
    // pushed harder where the lattice walls are bright.
    tex = mix(tex, 0.5 + 0.10 * (cfull + 0.5) * 0.0, 0.0);

    vec3 col = texture(texPrev, fract(tex + vec2(0.0, 0.0) * dd)).rgb;

    // Wall light: brightness falls off inside the cell.
    float wall = 1.0 - clamp(dd * 140.0, 0.0, 1.0);
    wall = smoothstep(0.0, 1.0, wall);
    // Life on the lattice follows mid/treble; keep a readable floor via rolloff.
    col += wall * (0.05 + 0.45 * u.mid + 0.35 * u.treble) * vec3(0.5, 0.7, 1.0)
         + 0.03 * u.rolloff;

    // Bass pulses the overall cell brightness.
    col *= 1.0 + 0.35 * u.bass + 0.30 * u.beat * clamp(dd * 300.0, 0.0, 1.0);

    float h = u.palette * 6.2831 + u.presetSeed * 0.7;
    float c = cos(h), sm = sin(h);
    mat3 hueRot = mat3(
        vec3(0.299 + 0.701*c + 0.168*sm, 0.587 - 0.587*c + 0.330*sm, 0.114 - 0.114*c - 0.497*sm),
        vec3(0.299 - 0.299*c - 0.328*sm, 0.587 + 0.413*c + 0.035*sm, 0.114 - 0.114*c + 0.292*sm),
        vec3(0.299 - 0.300*c + 1.250*sm, 0.587 - 0.588*c - 1.050*sm, 0.114 + 0.886*c - 0.203*sm)
    );
    vec3 shifted = hueRot * col.rgb;
    float luma = dot(shifted, vec3(0.299, 0.587, 0.114));
    shifted = mix(vec3(luma), shifted, 0.85);
    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    float energy = clamp(0.75*u.bass + 0.55*u.mid + 0.55*u.treble + 0.55*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    shifted = teeter_crossfade(shifted, tex, u.dissolve);
    outColor = vec4(shifted, 1.0);
}