#version 330 core

// Vertex Attributes
layout(location = 0) in vec3 position;
layout(location = NORMAL_LOC) in vec3 normal;
layout(location = INST_M_LOC) in mat4 inst_m;

// Uniforms
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

// Outputs
out vec3 frag_position;
out vec3 frag_normal;

void main()
{
    gl_Position = P * V * M * inst_m * vec4(position, 1);
    frag_position = vec3(M * inst_m * vec4(position, 1.0));

    mat4 N = transpose(inverse(M * inst_m));
    frag_normal = normalize(vec3(N * vec4(normal, 0.0)));

    mat4 invV = inverse(V);
    vec4 camPos_world = invV * vec4(0.0, 0.0, 0.0, 1.0);
    vec3 camera_position = camPos_world.xyz / camPos_world.w;

    vec3 view_vector = normalize(camera_position - frag_position);
    if(dot(view_vector, frag_normal) < 0.0) {
        frag_normal = -frag_normal;
    }
}