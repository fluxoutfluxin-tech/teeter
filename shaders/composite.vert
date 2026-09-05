#version 450

// Fullscreen triangle passes UV to the fragment stage.
// Draw with 3 vertices, no VAO buffer needed.

layout(location = 0) out vec2 uv;

void main() {
    // Triangle covering the whole NDC space.
    vec2 pos = vec2(
        (gl_VertexIndex == 2) ? 3.0 : -1.0,
        (gl_VertexIndex == 1) ? 3.0 : -1.0
    );
    gl_Position = vec4(pos, 0.0, 1.0);
    uv = 0.5 * (pos + 1.0);
}
