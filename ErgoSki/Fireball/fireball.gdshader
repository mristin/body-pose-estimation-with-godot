shader_type canvas_item;

uniform float time_scale = 1.0;
uniform vec4 core_color : source_color = vec4(1.0, 1.0, 0.7, 1.0);
uniform vec4 glow_color : source_color = vec4(1.0, 0.6, 0.2, 1.0);

void fragment() {
    vec2 uv = UV - vec2(0.5);
    float d = length(uv);

    // animated flicker
    float t = TIME * time_scale;
    float noise = sin(uv.x * 20.0 + t * 5.0) * 0.02;

    d += noise;

    // core + glow
    float core = smoothstep(0.2, 0.0, d);
    float glow = smoothstep(0.5, 0.2, d);

    vec3 color = glow_color.rgb * glow +
                 core_color.rgb * core;

    float alpha = glow * COLOR.a;

    COLOR = vec4(color, alpha);
}