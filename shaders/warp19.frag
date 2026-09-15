#version 450

// warp19 â€” infinite staircase (surreal).
//
// Feedback folds into rising rows that recede toward a vanishing point.
// Techno-escalator energy: rows climb with bass, roofline flickers on the
// beat, and the rolloff darkens the far end of the run while flux makes the
// treadmills slide.

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

    // Vanishing point at the top; rows recede downward. Pans up with bass.
    vec2 q = vec2(p.x, p.y + 0.5);
    float par = 1.0 - 0.75 * length(q);          // perspective falloff
    par = max(par, 0.001);

    // Rows climb with sub-bass; slide sideways on flux.
    float stepPix = 0.15 * (0.5 + u.subBass);
    float row = floor(q.y / stepPix);
    float ry = (q.y - row * stepPix) / stepPix;  // 0 = front..1 = back of row
    float slide = u.flux * 0.3 * sin(row * 1.7 + u.iTime * 1.4);
    float x = q.x / par + slide;

    // Speed the treads with the beat.
    float tread = fract(-u.iTime * (0.2 + 0.8 * u.beat) * (0.25 + 0.4 * u.bass) + row * 0.5);
    x += tread;

    vec2 tex = fract(vec2(x / u.aspect, ry) );
    vec3 col = texture(texPrev, tex).rgb;

    // Step edges: each row gets a moving light bar.
    float edge = smoothstep(0.0, 0.25, ry) * (1.0 - smoothstep(0.72, 1.0, ry));
    col += edge * (0.04 + 0.35 * u.treble) * vec3(1.0, 0.8, 0.6);
    // Riser glow on bass; beat pulses the whole row.
    col += ((1.0 - edge) * 0.25 + 0.6 * u.beat * edge) * 0.06 * u.bass * vec3(0.5, 0.4, 1.0);
    // Recede into dark toward the vanishing point (rolloff lights the far run).
    col *= (0.25 + 0.75 * ry) + 0.15 * (1.0 - u.rolloff);
    col *= 0.75 + 0.5 * u.bass;

    float h = u.palette * 6.2831 + u.presetSeed * 1.1;
    float c = cos(h), sm = sin(h);
    mat3 hueRot = mat3(
        vec3(0.299 + 0.701*c + 0.168*sm, 0.587 - 0.587*c + 0.330*sm, 0.114 - 0.114*c - 0.497*sm),
        vec3(0.299 - 0.299*c - 0.328*sm, 0.587 + 0.413*c + 0.035*sm, 0.114 - 0.114*c + 0.292*sm),
        vec3(0.299 - 0.300*c + 1.250*sm, 0.587 - 0.588*c - 1.050*sm, 0.114 + 0.886*c - 0.203*sm)
    );
    vec3 shifted = hueRot * col.rgb;
    float luma = dot(shifted, vec3(0.299, 0.587, 0.114));
    shifted = mix(vec3(luma), shifted, 0.84);
    shifted = max(shifted, col.rgb * 0.02 + vec3(0.004));

    float energy = clamp(0.85*u.bass + 0.45*u.mid + 0.55*u.treble + 0.60*u.beat, 0.0, 1.0);
    shifted = teeter_colorize(shifted, energy);
    shifted = teeter_crossfade(shifted, tex, u.dissolve);
    outColor = vec4(shifted, 1.0);
}