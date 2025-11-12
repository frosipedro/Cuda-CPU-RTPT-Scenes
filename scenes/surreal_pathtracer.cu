// surreal_pathtracer.cu
// Compile: nvcc surreal_pathtracer.cu -O2 -arch=sm_86 -o surreal_pathtracer
// If using Tesla T4 (WSL common), use -arch=sm_75 or omit -arch.
// Requires: stb_image_write.h in same folder (https://github.com/nothings/stb/blob/master/stb_image_write.h)

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>

inline void checkCuda(cudaError_t e, const char* msg){
    if (e != cudaSuccess){
        std::cerr << "CUDA error " << msg << ": " << cudaGetErrorString(e) << std::endl;
        exit(1);
    }
}

// ---------- small math types ----------
struct Vec3 {
    float x,y,z;
    __host__ __device__ Vec3():x(0),y(0),z(0){}
    __host__ __device__ Vec3(float a,float b,float c):x(a),y(b),z(c){}
    __host__ __device__ Vec3 operator+(const Vec3& b) const { return Vec3(x+b.x,y+b.y,z+b.z); }
    __host__ __device__ Vec3 operator-(const Vec3& b) const { return Vec3(x-b.x,y-b.y,z-b.z); }
    __host__ __device__ Vec3 operator*(const Vec3& b) const { return Vec3(x*b.x,y*b.y,z*b.z); }
    __host__ __device__ Vec3 operator*(float s) const { return Vec3(x*s,y*s,z*s); }
    __host__ __device__ Vec3 operator/(float s) const { return Vec3(x/s,y/s,z/s); }
    __host__ __device__ float dot(const Vec3& b) const { return x*b.x + y*b.y + z*b.z; }
    __host__ __device__ Vec3 normalized() const { float l = sqrtf(x*x+y*y+z*z); return (l>0)?(*this)/l:Vec3(0,0,0); }
    __host__ __device__ float length() const { return sqrtf(x*x+y*y+z*z); }
};
__host__ __device__ inline Vec3 cross(const Vec3& a, const Vec3& b){
    return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__host__ __device__ inline Vec3 minv(const Vec3& a, const Vec3& b){
    return Vec3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
__host__ __device__ inline Vec3 maxv(const Vec3& a, const Vec3& b){
    return Vec3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

// ---------- ray, sphere, material ----------
struct Ray { Vec3 o,d; __host__ __device__ Ray(){} __host__ __device__ Ray(const Vec3& oo,const Vec3& dd):o(oo),d(dd){} __host__ __device__ Vec3 at(float t) const { return o + d * t; } };

enum MatType { DIFFUSE=0, METAL=1, DIELECTRIC=2, EMISSIVE=3 };

struct Material {
    Vec3 albedo;
    float fuzz;        // metal roughness
    float ref_idx;     // dielectric RI
    MatType type;
    Vec3 emission;
};

struct Sphere {
    Vec3 center;
    float radius;
    int mat_idx;
};

// ---------- device RNG ----------
__device__ float rndf(curandState* s){ return curand_uniform(s); }

__device__ Vec3 random_in_unit_sphere(curandState* st){
    while(true){
        float x = rndf(st)*2.0f - 1.0f;
        float y = rndf(st)*2.0f - 1.0f;
        float z = rndf(st)*2.0f - 1.0f;
        Vec3 p(x,y,z);
        if (p.dot(p) < 1.0f) return p;
    }
}

__device__ Vec3 random_unit_vector(curandState* st){
    float a = rndf(st) * 2.0f * 3.14159265359f;
    float z = rndf(st) * 2.0f - 1.0f;
    float r = sqrtf(1.0f - z*z);
    return Vec3(r*cosf(a), r*sinf(a), z);
}

__device__ Vec3 reflect(const Vec3& v, const Vec3& n){ return v - n*(2.0f * v.dot(n)); }

__device__ bool refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted){
    Vec3 uv = v.normalized();
    float dt = uv.dot(n);
    float disc = 1.0f - ni_over_nt*ni_over_nt*(1 - dt*dt);
    if (disc > 0.0f){
        refracted = (uv - n*dt) * ni_over_nt - n * sqrtf(disc);
        return true;
    }
    return false;
}

__device__ float schlick(float cosine, float ref_idx){
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f - r0)*powf((1 - cosine),5);
}

// ---------- sphere intersection ----------
__device__ bool hit_sphere(const Sphere& s, const Ray& r, float t_min, float t_max, float& t_hit, Vec3& p, Vec3& normal){
    Vec3 oc = r.o - s.center;
    float a = r.d.dot(r.d);
    float b = oc.dot(r.d);
    float c = oc.dot(oc) - s.radius*s.radius;
    float disc = b*b - a*c;
    if (disc > 0.0f){
        float sq = sqrtf(disc);
        float temp = (-b - sq) / a;
        if (temp < t_max && temp > t_min){
            t_hit = temp; p = r.at(t_hit); normal = (p - s.center) / s.radius; return true;
        }
        temp = (-b + sq) / a;
        if (temp < t_max && temp > t_min){
            t_hit = temp; p = r.at(t_hit); normal = (p - s.center) / s.radius; return true;
        }
    }
    return false;
}

// ---------- scene tracing (device) ----------
__device__ bool scene_hit(Sphere* spheres, int nspheres, const Ray& r, float t_min, float t_max, float& t_hit, Vec3& p, Vec3& normal, int& mat_idx){
    float closest = t_max;
    bool hit_any = false;
    for (int i=0;i<nspheres;i++){
        float th; Vec3 tp, tn;
        if (hit_sphere(spheres[i], r, t_min, closest, th, tp, tn)){
            closest = th; t_hit = th; p = tp; normal = tn; mat_idx = spheres[i].mat_idx; hit_any = true;
        }
    }
    return hit_any;
}

// ---------- path tracing integrator (device) ----------
__device__ Vec3 trace_path(Ray r, Sphere* spheres, int nspheres, Material* mats, curandState* st, int maxDepth, Vec3 env_color, float fog_density){
    Vec3 throughput(1,1,1);
    Vec3 accum(0,0,0);

    for (int depth=0; depth<maxDepth; ++depth){
        float t_hit; Vec3 hit_p, normal; int mat_i=-1;
        if (!scene_hit(spheres, nspheres, r, 0.001f, 1e20f, t_hit, hit_p, normal, mat_i)){
            // environment (simple HDR-ish)
            Vec3 unit = r.d.normalized();
            float t = 0.5f*(unit.y + 1.0f);
            Vec3 bg = env_color * t + Vec3(0.02f,0.02f,0.02f)*(1.0f - t);
            // apply thin fog (exponential)
            float dist = 1e6f; // no hit
            float fog = expf(-fog_density * dist);
            accum = accum + throughput * bg * fog;
            break;
        }

        Material mat = mats[mat_i];
        // emissive
        if (mat.type == EMISSIVE){
            accum = accum + throughput * mat.emission;
            break;
        }

        // scattering by material
        if (mat.type == DIFFUSE){
            // cosine-weighted hemisphere approx with random_unit_vector
            Vec3 target = hit_p + normal + random_unit_vector(st);
            r = Ray(hit_p, (target - hit_p).normalized());
            throughput = throughput * mat.albedo;
            continue;
        } else if (mat.type == METAL){
            Vec3 reflected = reflect(r.d.normalized(), normal);
            r = Ray(hit_p, (reflected + random_in_unit_sphere(st) * mat.fuzz).normalized());
            throughput = throughput * mat.albedo;
            if (r.d.dot(normal) <= 0.0f) break;
            continue;
        } else if (mat.type == DIELECTRIC){
            Vec3 outward_normal;
            Vec3 reflected = reflect(r.d, normal);
            float ni_over_nt;
            Vec3 refracted;
            float reflect_prob;
            float cosine;
            if (r.d.dot(normal) > 0.0f){
                outward_normal = normal * -1.0f;
                ni_over_nt = mat.ref_idx;
                cosine = mat.ref_idx * r.d.dot(normal) / r.d.length();
            } else {
                outward_normal = normal;
                ni_over_nt = 1.0f / mat.ref_idx;
                cosine = -r.d.dot(normal) / r.d.length();
            }
            if (refract(r.d, outward_normal, ni_over_nt, refracted)){
                reflect_prob = schlick(cosine, mat.ref_idx);
            } else reflect_prob = 1.0f;
            if (rndf(st) < reflect_prob){
                r = Ray(hit_p, reflected.normalized());
            } else {
                r = Ray(hit_p, refracted.normalized());
            }
            // dielectrics attenuate little, but we can keep throughput
            throughput = throughput * mat.albedo;
            continue;
        } else {
            break;
        }
    }
    return accum;
}

// ---------- curand init ----------
__global__ void init_rand(curandState* states, int width, int height, unsigned long seed){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x>=width || y>=height) return;
    int idx = y*width + x;
    curand_init(seed + idx, 0, 0, &states[idx]);
}

// ---------- render kernel ----------
__global__ void render_kernel(Vec3* fb, int width, int height, Sphere* spheres, int nspheres, Material* mats, curandState* states, int samples, int maxDepth, Vec3 cam_pos, Vec3 cam_forward, Vec3 cam_up, float fov, Vec3 env_color, float fog_density){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x>=width || y>=height) return;
    int idx = y*width + x;
    curandState local = states[idx];
    Vec3 color(0,0,0);

    Vec3 right = cross(cam_forward, cam_up).normalized();
    Vec3 up = cross(right, cam_forward).normalized();
    float aspect = float(width)/float(height);
    float scale = tanf( fov * 0.5f * 3.14159265f / 180.0f );

    for (int s=0; s<samples; ++s){
        float rx = rndf(&local);
        float ry = rndf(&local);
        float u = ( (x + rx) / float(width) - 0.5f ) * 2.0f * aspect * scale;
        float v = ( (y + ry) / float(height) - 0.5f ) * 2.0f * scale;
        Vec3 dir = (cam_forward + right * u + up * v).normalized();
        Ray r(cam_pos, dir);
        color = color + trace_path(r, spheres, nspheres, mats, &local, maxDepth, env_color, fog_density);
    }

    states[idx] = local;
    color = color / float(samples);
    // tone mapping & gamma approximate
    color = Vec3( color.x / (1.0f + color.x), color.y / (1.0f + color.y), color.z / (1.0f + color.z) );
    color = Vec3( sqrtf(color.x), sqrtf(color.y), sqrtf(color.z) );
    fb[idx] = color;
}

// ---------- host helpers: procedural scene ----------
void make_surreal_scene(std::vector<Sphere>& spheres, std::vector<Material>& mats){
    mats.clear();
    spheres.clear();

    // ---------- MATERIALS ----------
    Material white; white.albedo=Vec3(0.8f,0.8f,0.8f); white.type=DIFFUSE;
    Material red; red.albedo=Vec3(0.9f,0.1f,0.1f); red.type=DIFFUSE;
    Material green; green.albedo=Vec3(0.1f,0.9f,0.2f); green.type=DIFFUSE;
    Material metal; metal.albedo=Vec3(0.9f,0.9f,0.95f); metal.type=METAL; metal.fuzz=0.02f;
    Material blackMirror; blackMirror.albedo=Vec3(0.02f,0.02f,0.02f); blackMirror.type=METAL; blackMirror.fuzz=0.0f;
    Material glass; glass.albedo=Vec3(0.98f,0.98f,1.0f); glass.type=DIELECTRIC; glass.ref_idx=1.45f;
    Material light; light.type=EMISSIVE; light.emission=Vec3(6.0f,6.0f,6.0f);

    mats.push_back(white);       //0
    mats.push_back(red);         //1
    mats.push_back(green);       //2
    mats.push_back(metal);       //3
    mats.push_back(blackMirror); //4
    mats.push_back(glass);       //5
    mats.push_back(light);       //6

    // ---------- BOX WALLS (BIG SPHERES) ----------
    float R = 1000.0f;

    // Floor
    spheres.push_back({Vec3(0, -R-1.0f, -3.5f), R, 0});

    // Ceiling light (emissive)
    spheres.push_back({Vec3(0, R+5.5f, -3.5f), R, 6});

    // Back wall (mirror black)
    spheres.push_back({Vec3(0, 0, -R-8.0f), R, 4});

    // Left red wall
    spheres.push_back({Vec3(-R-3.0f, 1.5f, -3.5f), R, 1});

    // Right green wall
    spheres.push_back({Vec3(R+3.0f, 1.5f, -3.5f), R, 2});


    // ---------- SCULPTURE: 3 large spheres ----------
    spheres.push_back({Vec3(-1.0f, 1.0f, -4.5f), 1.0f, 3});
    spheres.push_back({Vec3(1.4f, 1.2f, -4.2f), 0.85f, 5});
    spheres.push_back({Vec3(0.4f, 0.5f, -3.0f), 0.55f, 3});

    // ---------- FLOATING GLOW RINGS ----------
    for(int i=0;i<5;i++){
        float x = sinf(i*1.3f)*1.8f;
        float y = 2.0f + cosf(i*1.3f)*0.25f;
        float z = -3.8f - i*0.4f;
        spheres.push_back({Vec3(x,y,z), 0.25f, 6});
    }
}

// ---------- host main ----------
int main(){
    std::cout << "Surreal Path Tracer (CUDA)\n";
    std::cout << "This scene is procedural-heavy and can be slow. Use small res/samples for testing.\n";

    int width = 1280, height = 720;
    std::cout << "Enter resolution (width height, e.g. 1920 1080): ";
    std::cin >> width >> height;

    int samples = 100;
    std::cout << "Samples per pixel (e.g. 32 quick, 100+ nicer): "; std::cin >> samples;
    int maxDepth = 6;
    std::cout << "Max bounces (depth, e.g. 4-8): "; std::cin >> maxDepth;

    std::cout << "Fog density (0.0 none, 0.02 light): ";
    float fog_density; std::cin >> fog_density;

    // build scene host-side
    std::vector<Sphere> spheres;
    std::vector<Material> mats;
    make_surreal_scene(spheres, mats);
    int nspheres = (int)spheres.size();
    size_t npixels = (size_t)width * height;
    std::cout << "Scene: spheres=" << nspheres << "  mats=" << mats.size() << "\n";

    // allocate framebuffer
    Vec3* fb = new Vec3[npixels];

    // copy scene to device
    Sphere* d_spheres; Material* d_mats; Vec3* d_fb; curandState* d_states;
    checkCuda(cudaMalloc(&d_spheres, sizeof(Sphere) * nspheres), "alloc spheres");
    checkCuda(cudaMalloc(&d_mats, sizeof(Material) * mats.size()), "alloc mats");
    checkCuda(cudaMalloc(&d_fb, sizeof(Vec3) * npixels), "alloc fb");
    checkCuda(cudaMalloc(&d_states, sizeof(curandState) * npixels), "alloc rand");

    checkCuda(cudaMemcpy(d_spheres, spheres.data(), sizeof(Sphere) * nspheres, cudaMemcpyHostToDevice), "copy spheres");
    checkCuda(cudaMemcpy(d_mats, mats.data(), sizeof(Material) * mats.size(), cudaMemcpyHostToDevice), "copy mats");

    // camera
    Vec3 cam_pos(0.0f, 1.2f, 3.0f);
    Vec3 cam_look(0.0f, 1.0f, -3.5f);
    Vec3 cam_forward = (cam_look - cam_pos).normalized();
    Vec3 cam_up(0.0f, 1.0f, 0.0f);
    float fov = 50.0f;

    // environment color (can be HDR like)
    Vec3 env_color(0.8f, 0.95f, 1.0f);

    dim3 threads(16,16);
    dim3 blocks( (width + threads.x -1)/threads.x, (height + threads.y -1)/threads.y );

    init_rand<<<blocks, threads>>>(d_states, width, height, (unsigned long) time(NULL));
    checkCuda(cudaGetLastError(), "init_rand kernel");
    checkCuda(cudaDeviceSynchronize(), "sync after init_rand");

    std::cout << "Rendering (GPU) ...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    render_kernel<<<blocks, threads>>>(d_fb, width, height, d_spheres, nspheres, d_mats, d_states, samples, maxDepth, cam_pos, cam_forward, cam_up, fov, env_color, fog_density);
    checkCuda(cudaGetLastError(), "render_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync after render");

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "GPU render time: " << dt.count() << " s\n";

    // copy result
    checkCuda(cudaMemcpy(fb, d_fb, sizeof(Vec3) * npixels, cudaMemcpyDeviceToHost), "copy fb to host");

    // save PNG
    unsigned char* out = new unsigned char[npixels * 3];
    for (size_t i=0;i<npixels;i++){
        Vec3 c = fb[i];
        // clamp
        float r = fminf(fmaxf(c.x, 0.0f), 1.0f);
        float g = fminf(fmaxf(c.y, 0.0f), 1.0f);
        float b = fminf(fmaxf(c.z, 0.0f), 1.0f);
        out[3*i+0] = (unsigned char)(r * 255.0f);
        out[3*i+1] = (unsigned char)(g * 255.0f);
        out[3*i+2] = (unsigned char)(b * 255.0f);
    }
    std::string name = "surreal_path.png";
    stbi_write_png(name.c_str(), width, height, 3, out, width*3);
    std::cout << "Saved: " << name << "\n";

    // cleanup
    delete[] out;
    delete[] fb;
    cudaFree(d_spheres); cudaFree(d_mats); cudaFree(d_fb); cudaFree(d_states);

    std::cout << "Done.\n";
    return 0;
}
