#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

static int prompt_int(const std::string& message, int default_value) {
    for (;;) {
        std::cout << "\n" << message;
        std::string line;
        if (!std::getline(std::cin, line) || line.empty()) {
            return default_value;
        }

        std::stringstream parser(line);
        int value;
        char extra;
        if (parser >> value && !(parser >> extra)) {
            return value;
        }

        std::cout << "Please enter a whole number or press Enter to use the default value.\n";
    }
}

static void ensure_output_directory() {
#ifdef _WIN32
    _mkdir("images");
#else
    mkdir("images", 0755);
#endif
}

static std::string timestamp_suffix() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};

#ifdef _WIN32
    localtime_s(&local_time, &now_time);
#else
    localtime_r(&now_time, &local_time);
#endif

    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &local_time);
    return buffer;
}

static std::string make_output_name(const std::string& scene_name, const std::string& mode_name, const std::string& stamp) {
    return std::string("images/") + scene_name + "_" + mode_name + "_" + stamp + ".png";
}

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

#define MAX_SPHERES 8
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

__host__ __device__ Vec3 background_color(const Vec3& rd) {
    Vec3 unit = rd.normalize();
    float t = 0.5f * (unit.y + 1.0f);
    Vec3 sky = Vec3(0.22f, 0.30f, 0.46f) * (1.0f - t) + Vec3(0.82f, 0.90f, 1.0f) * t;
    Vec3 sun_dir = Vec3(-0.35f, 0.75f, -0.55f).normalize();
    float sun = powf(fmaxf(0.0f, unit.dot(sun_dir)), 256.0f);
    return sky + Vec3(1.0f, 0.86f, 0.62f) * sun * 1.5f;
}

__host__ Vec3 ray_color(const Vec3& ro, const Vec3& rd, const Sphere* spheres, int sphere_count, const Light* lights, int light_count, int depth) {
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

    return background_color(rd);
}

__device__ Vec3 ray_color_gpu(const Vec3& ro, const Vec3& rd, const Sphere* spheres, int sphere_count, const Light* lights, int light_count) {
    Vec3 current_ro = ro;
    Vec3 current_rd = rd;
    Vec3 color(0, 0, 0);
    Vec3 throughput(1, 1, 1);

    for (int depth = 0; depth <= MAX_BOUNCES; depth++) {
        float t_min;
        Vec3 hit_point, normal;
        int sphere_idx = find_closest_sphere(current_ro, current_rd, spheres, sphere_count, t_min, hit_point, normal);

        if (sphere_idx >= 0) {
            Vec3 direct(0, 0, 0);
            const Sphere& sphere = spheres[sphere_idx];

            for (int l = 0; l < light_count; l++) {
                if (!in_shadow(hit_point, lights[l].pos, spheres, sphere_count)) {
                    Vec3 to_light = (lights[l].pos - hit_point).normalize();
                    float diff = fmaxf(0.0f, normal.dot(to_light));

                    Vec3 view = (current_ro - hit_point).normalize();
                    Vec3 half = (to_light + view).normalize();
                    float spec = powf(fmaxf(0.0f, normal.dot(half)), sphere.shininess);

                    direct = direct + sphere.color * diff * lights[l].color * 0.7f + Vec3(1, 1, 1) * spec * lights[l].color * 0.3f;
                }
            }

            color = color + throughput * direct;

            if (depth == MAX_BOUNCES) {
                break;
            }

            current_rd = current_rd - normal * 2.0f * current_rd.dot(normal);
            current_ro = hit_point;
            throughput = throughput * sphere.shininess * 0.1f;
            continue;
        }

        if (current_rd.y < 0) {
            float tplane = -current_ro.y / current_rd.y;
            if (tplane > 0.001f) {
                Vec3 p = current_ro + current_rd * tplane;
                int checker = (int(floorf(p.x * 2) + floorf(p.z * 2)) & 1);
                Vec3 col = checker ? Vec3(0.9, 0.9, 0.9) : Vec3(0.1, 0.1, 0.1);

                bool shadowed = false;
                for (int l = 0; l < light_count; l++) {
                    if (in_shadow(p, lights[l].pos, spheres, sphere_count)) {
                        shadowed = true;
                        break;
                    }
                }

                color = color + throughput * col * (shadowed ? 0.4f : 0.8f);
            } else {
                color = color + throughput * background_color(current_rd);
            }
        } else {
            color = color + throughput * background_color(current_rd);
        }

        break;
    }

    return color;
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
            color = color + ray_color_gpu(cam_pos, rd, spheres, sphere_count, lights, light_count);
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
    int width = 1280, height = 720;
    size_t fb_size = 0;
    const std::string render_stamp = timestamp_suffix();
    ensure_output_directory();

    Vec3 cam_pos(0.4f, 2.0f, 5.5f);
    Vec3 cam_look(0.0f, 0.75f, -2.2f);
    
    // Richer RT scene with mixed materials and more depth.
    Sphere spheres[MAX_SPHERES];
    int sphere_count = 8;
    spheres[0] = { Vec3(0.0f, 1.0f, -3.2f), 1.0f, Vec3(1.0f, 0.2f, 0.2f), 32.0f };
    spheres[1] = { Vec3(-2.7f, 1.0f, -4.0f), 0.82f, Vec3(0.2f, 0.9f, 0.35f), 18.0f };
    spheres[2] = { Vec3(2.6f, 1.0f, -4.1f), 0.82f, Vec3(0.2f, 0.45f, 1.0f), 18.0f };
    spheres[3] = { Vec3(-1.1f, 0.5f, -1.5f), 0.5f, Vec3(1.0f, 0.95f, 0.25f), 72.0f };
    spheres[4] = { Vec3(1.6f, 0.75f, -2.0f), 0.72f, Vec3(1.0f, 0.25f, 0.95f), 42.0f };
    spheres[5] = { Vec3(-3.0f, 0.55f, -2.3f), 0.55f, Vec3(0.9f, 0.6f, 0.2f), 14.0f };
    spheres[6] = { Vec3(3.1f, 0.82f, -2.8f), 0.82f, Vec3(0.92f, 0.92f, 0.97f), 96.0f };
    spheres[7] = { Vec3(0.0f, 2.35f, -2.0f), 0.35f, Vec3(0.15f, 0.85f, 0.95f), 54.0f };
    
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

    std::cout << "Advanced Ray Tracer\n";
    std::cout << "RT-only scene with direct lighting, reflections, a richer composition, and a cinematic sky.\n";
    std::cout << "Default resolution is 1280x720, with presets up to 2560x1440. Output files are written to images/.\n";
    std::cout << "\nRender setup will be summarized after you answer the prompts.\n";

    int option;
    do {
        option = prompt_int("Choose a render mode [1=CPU, 2=GPU, 3=compare both] [Enter=2]: ", 2);
        if (option < 1 || option > 3) {
            std::cout << "Please choose 1, 2, or 3.\n";
        }
    } while (option < 1 || option > 3);

    const char* mode_label = (option == 1) ? "CPU only" : (option == 2 ? "GPU only" : "CPU + GPU comparison");
    int preset = prompt_int("Resolution preset [1=1280x720, 2=1920x1080, 3=2560x1440] [Enter=1]: ", 1);
    while (preset < 1 || preset > 3) {
        std::cout << "Please choose 1, 2, or 3.\n";
        preset = prompt_int("Resolution preset [1=1280x720, 2=1920x1080, 3=2560x1440] [Enter=1]: ", 1);
    }

    if (preset == 2) { width = 1920; height = 1080; }
    else if (preset == 3) { width = 2560; height = 1440; }

    fb_size = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(Vec3);
    Vec3* fb = new Vec3[static_cast<size_t>(width) * static_cast<size_t>(height)];

    std::cout << "\nRender setup:\n";
    std::cout << "  Mode: " << mode_label << "\n";
    std::cout << "  Resolution: " << width << "x" << height << "\n";
    std::cout << "  Output folder: images/\n\n";
    // ================= CPU =================
    if (option == 1 || option == 3) {
        auto start_cpu = std::chrono::high_resolution_clock::now();
        render_cpu(fb, width, height, cam_pos, spheres, sphere_count, lights, light_count);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> dur_cpu = end_cpu - start_cpu;
        std::cout << "CPU time: " << dur_cpu.count() << "s\n";

        std::string cpu_output = make_output_name("simple_raytracer", "cpu", render_stamp);
        save_png(cpu_output.c_str(), fb, width, height);
        std::cout << "Saved image: " << cpu_output << "\n";
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

        std::string gpu_output = make_output_name("simple_raytracer", "gpu", render_stamp);
        save_png(gpu_output.c_str(), fb, width, height);
        std::cout << "Saved image: " << gpu_output << "\n";
    }

    delete[] fb;
    return 0;
}
