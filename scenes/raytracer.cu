#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector>

// ---------------- VETORES E GEOMETRIA ----------------
struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float a, float b, float c) : x(a), y(b), z(c) {}

    __host__ __device__ Vec3 operator+(const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    __host__ __device__ Vec3 operator-(const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    __host__ __device__ Vec3 operator*(float b) const { return Vec3(x * b, y * b, z * b); }
    __host__ __device__ Vec3 operator*(const Vec3& b) const { return Vec3(x * b.x, y * b.y, z * b.z); }

    __host__ __device__ float dot(const Vec3& b) const { return x * b.x + y * b.y + z * b.z; }
    __host__ __device__ Vec3 normalize() const {
        float len = sqrtf(x*x + y*y + z*z);
        return Vec3(x / len, y / len, z / len);
    }
    __host__ __device__ Vec3 cross(const Vec3& b) const {
        return Vec3(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
    }
};

// ---------------- CENA ----------------
struct Sphere { Vec3 center; float radius; Vec3 color; float shininess; };
struct Light  { Vec3 pos; Vec3 color; };

#define MAX_SPHERES 5
#define MAX_LIGHTS 8
#define MAX_BOUNCES 8
#define AA_SAMPLES 4

__host__ __device__ bool hit_sphere(const Sphere& s, const Vec3& ro, const Vec3& rd, float& t) {
    Vec3 oc = ro - s.center;
    float b = 2.0f * oc.dot(rd);
    float c = oc.dot(oc) - s.radius * s.radius;
    float discriminant = b*b - 4*c;
    if (discriminant < 0) return false;
    t = (-b - sqrtf(discriminant)) / 2.0f;
    return t > 0.001f;
}

__host__ __device__ int find_closest_sphere(const Vec3& ro, const Vec3& rd, const Sphere* spheres, int count, float& t_min, Vec3& hit_point, Vec3& normal) {
    t_min = 1e9f;
    int closest = -1;
    for (int i = 0; i < count; i++) {
        float t;
        if (hit_sphere(spheres[i], ro, rd, t) && t < t_min) {
            t_min = t;
            closest = i;
            hit_point = ro + rd * t;
            normal = (hit_point - spheres[i].center).normalize();
        }
    }
    return closest;
}

__host__ __device__ bool in_shadow(const Vec3& p, const Vec3& light_pos, const Sphere* spheres, int count) {
    Vec3 to_light = (light_pos - p).normalize();
    float dist_to_light = (light_pos - p).dot(light_pos - p);
    
    for (int i = 0; i < count; i++) {
        float t;
        if (hit_sphere(spheres[i], p, to_light, t)) {
            if (t * t < dist_to_light) return true;
        }
    }
    return false;
}

__host__ __device__ Vec3 ray_color(const Vec3& ro, const Vec3& rd, const Sphere* spheres, int sphere_count, const Light* lights, int light_count, int depth) {
    if (depth > MAX_BOUNCES) return Vec3(0, 0, 0);

    float t_min;
    Vec3 hit_point, normal;
    int sphere_idx = find_closest_sphere(ro, rd, spheres, sphere_count, t_min, hit_point, normal);

    if (sphere_idx >= 0) {
        Vec3 color(0, 0, 0);
        const Sphere& sphere = spheres[sphere_idx];

        // Difuso + Especular com múltiplas luzes
        for (int l = 0; l < light_count; l++) {
            if (!in_shadow(hit_point, lights[l].pos, spheres, sphere_count)) {
                Vec3 to_light = (lights[l].pos - hit_point).normalize();
                float diff = fmaxf(0.0f, normal.dot(to_light));
                
                Vec3 view = (ro - hit_point).normalize();
                Vec3 half = (to_light + view).normalize();
                float spec = powf(fmaxf(0.0f, normal.dot(half)), sphere.shininess);
                
                color = color + sphere.color * diff * lights[l].color * 0.7f + Vec3(1, 1, 1) * spec * lights[l].color * 0.3f;
            }
        }

        // Reflexo
        Vec3 reflected = rd - normal * 2.0f * rd.dot(normal);
        Vec3 reflection = ray_color(hit_point, reflected, spheres, sphere_count, lights, light_count, depth + 1);
        color = color + reflection * sphere.shininess * 0.1f;

        return color;
    }

    // Plano com mais detalhes
    if (rd.y < 0) {
        float tplane = -ro.y / rd.y;
        if (tplane > 0.001f) {
            Vec3 p = ro + rd * tplane;
            int checker = (int(floorf(p.x * 2) + floorf(p.z * 2)) & 1);
            Vec3 col = checker ? Vec3(0.9, 0.9, 0.9) : Vec3(0.1, 0.1, 0.1);
            
            // Sombra do plano
            bool shadowed = false;
            for (int l = 0; l < light_count; l++) {
                if (in_shadow(p, lights[l].pos, spheres, sphere_count)) {
                    shadowed = true;
                    break;
                }
            }
            return col * (shadowed ? 0.4f : 0.8f);
        }
    }

    return Vec3(0.5, 0.7, 1.0);
}

// ---------------- GPU ----------------
__global__ void render_gpu(Vec3* fb, int width, int height, Vec3 cam_pos, const Sphere* spheres, int sphere_count, const Light* lights, int light_count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    Vec3 color(0, 0, 0);
    for (int sy = 0; sy < AA_SAMPLES; sy++) {
        for (int sx = 0; sx < AA_SAMPLES; sx++) {
            float u = (2.0f * (x + sx / (float)AA_SAMPLES) / width - 1.0f);
            float v = (2.0f * (y + sy / (float)AA_SAMPLES) / height - 1.0f);
            Vec3 rd = Vec3(u, -v, -1.5).normalize();
            color = color + ray_color(cam_pos, rd, spheres, sphere_count, lights, light_count, 0);
        }
    }
    fb[y * width + x] = color * (1.0f / (AA_SAMPLES * AA_SAMPLES));
}

// ---------------- CPU ----------------
void render_cpu(Vec3* fb, int width, int height, Vec3 cam_pos, const Sphere* spheres, int sphere_count, const Light* lights, int light_count) {
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            Vec3 color(0, 0, 0);
            for (int sy = 0; sy < AA_SAMPLES; sy++) {
                for (int sx = 0; sx < AA_SAMPLES; sx++) {
                    float u = (2.0f * (x + sx / (float)AA_SAMPLES) / width - 1.0f);
                    float v = (2.0f * (y + sy / (float)AA_SAMPLES) / height - 1.0f);
                    Vec3 rd = Vec3(u, -v, -1.5).normalize();
                    color = color + ray_color(cam_pos, rd, spheres, sphere_count, lights, light_count, 0);
                }
            }
            fb[y * width + x] = color * (1.0f / (AA_SAMPLES * AA_SAMPLES));
        }
}

__host__ void save_png(const char* path, const Vec3* fb, int width, int height) {
    std::vector<unsigned char> img(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        Vec3 c = fb[i];
        img[3 * i + 0] = static_cast<unsigned char>(fminf(fmaxf(c.x, 0.0f), 1.0f) * 255.99f);
        img[3 * i + 1] = static_cast<unsigned char>(fminf(fmaxf(c.y, 0.0f), 1.0f) * 255.99f);
        img[3 * i + 2] = static_cast<unsigned char>(fminf(fmaxf(c.z, 0.0f), 1.0f) * 255.99f);
    }
    stbi_write_png(path, width, height, 3, img.data(), width * 3);
}

// ---------------- MAIN ----------------
int main() {
    int width = 3840, height = 2880;
    size_t fb_size = width * height * sizeof(Vec3);

    Vec3* fb = new Vec3[width * height];
    Vec3 cam_pos(0, 2, 5);
    
    // Múltiplas esferas - 5 esferas
    Sphere spheres[MAX_SPHERES];
    int sphere_count = 5;
    spheres[0] = { Vec3(0, 1, -3), 1.0f, Vec3(1, 0, 0), 32.0f };
    spheres[1] = { Vec3(-2.5, 1, -4), 0.8f, Vec3(0, 1, 0), 16.0f };
    spheres[2] = { Vec3(2.5, 1, -4), 0.8f, Vec3(0, 0, 1), 16.0f };
    spheres[3] = { Vec3(-1, 0.5, -1.5), 0.5f, Vec3(1, 1, 0), 64.0f };
    spheres[4] = { Vec3(1.5, 0.7, -2), 0.7f, Vec3(1, 0, 1), 32.0f };
    
    // Múltiplas luzes - agora com 8
    Light lights[MAX_LIGHTS];
    int light_count = 8;
    lights[0] = { Vec3(5, 10, 5), Vec3(1, 1, 1) };
    lights[1] = { Vec3(-5, 8, 3), Vec3(0.6, 0.6, 1) };
    lights[2] = { Vec3(0, 5, 8), Vec3(1, 0.6, 0.6) };
    lights[3] = { Vec3(8, 6, -5), Vec3(0.8, 1, 0.5) };
    lights[4] = { Vec3(-8, 7, -3), Vec3(1, 0.8, 0.8) };
    lights[5] = { Vec3(3, 9, -8), Vec3(0.5, 1, 0.8) };
    lights[6] = { Vec3(-3, 8, 0), Vec3(1, 0.5, 1) };
    lights[7] = { Vec3(0, 12, 0), Vec3(0.9, 0.9, 0.9) };

    std::cout << "Escolha o modo de execução:\n";
    std::cout << "1 - CPU somente\n";
    std::cout << "2 - GPU somente\n";
    std::cout << "3 - Comparar ambos\n> ";
    int option;
    std::cin >> option;

    // ================= CPU =================
    if (option == 1 || option == 3) {
        auto start_cpu = std::chrono::high_resolution_clock::now();
        render_cpu(fb, width, height, cam_pos, spheres, sphere_count, lights, light_count);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> dur_cpu = end_cpu - start_cpu;
        std::cout << "CPU time: " << dur_cpu.count() << "s\n";

        save_png("output_cpu.png", fb, width, height);
        std::cout << "Imagem CPU salva em output_cpu.png\n";
    }

    // ================= GPU =================
    if (option == 2 || option == 3) {
        Vec3 *d_fb;
        Sphere *d_spheres;
        Light *d_lights;
        
        cudaMalloc(&d_fb, fb_size);
        cudaMalloc(&d_spheres, sizeof(Sphere) * sphere_count);
        cudaMalloc(&d_lights, sizeof(Light) * light_count);
        
        cudaMemcpy(d_spheres, spheres, sizeof(Sphere) * sphere_count, cudaMemcpyHostToDevice);
        cudaMemcpy(d_lights, lights, sizeof(Light) * light_count, cudaMemcpyHostToDevice);
        
        cudaDeviceSetLimit(cudaLimitStackSize, 8192);
        
        dim3 threads(16, 16);
        dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        render_gpu<<<blocks, threads>>>(d_fb, width, height, cam_pos, d_spheres, sphere_count, d_lights, light_count);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "GPU time: " << ms / 1000.0f << "s\n";

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
        }

        cudaMemcpy(fb, d_fb, fb_size, cudaMemcpyDeviceToHost);
        cudaFree(d_fb);
        cudaFree(d_spheres);
        cudaFree(d_lights);

        save_png("output_gpu.png", fb, width, height);
        std::cout << "Imagem GPU salva em output_gpu.png\n";
    }

    delete[] fb;
    return 0;
}
