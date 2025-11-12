// render_all.cu
// Compile: nvcc render_all.cu -O2 -arch=sm_86 -o render_all
// Ajuste -arch conforme sua GPU (por ex. sm_75 para T4/older, sm_86 para 30xx, sm_80/86 etc).
// Requires: stb_image_write.h in same folder (https://github.com/nothings/stb/blob/master/stb_image_write.h)

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <omp.h>

inline void checkCuda(cudaError_t e, const char* msg){
    if (e != cudaSuccess){ std::cerr << "CUDA error " << msg << ": " << cudaGetErrorString(e) << std::endl; exit(1); }
}

// -------------------------- Basic Vec3 --------------------------
struct Vec3 {
    float x,y,z;
    __host__ __device__ Vec3():x(0),y(0),z(0){}
    __host__ __device__ Vec3(float a, float b, float c):x(a),y(b),z(c){}
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

// -------------------------- Ray & Sphere --------------------------
struct Ray { Vec3 o,d; __host__ __device__ Ray(){} __host__ __device__ Ray(const Vec3& oo,const Vec3& dd):o(oo),d(dd){} __host__ __device__ Vec3 at(float t) const { return o + d * t; } };

struct Sphere {
    Vec3 center;
    float radius;
    int mat; // index into materials
};

// Material types
enum MatType { DIFFUSE=0, METAL=1, EMISSIVE=2 };
struct Material {
    Vec3 albedo;
    float fuzz;
    MatType type;
    Vec3 emission;
};

// -------------------------- Simple Ray Tracer (CPU & GPU) --------------------------
// CPU helpers
bool hit_sphere_cpu(const Sphere& s, const Ray& r, float t_min, float t_max, float &t_hit, Vec3 &hit_p, Vec3 &normal){
    Vec3 oc = r.o - s.center;
    float a = r.d.dot(r.d);
    float b = oc.dot(r.d);
    float c = oc.dot(oc) - s.radius*s.radius;
    float disc = b*b - a*c;
    if (disc > 0){
        float sq = sqrtf(disc);
        float temp = (-b - sq) / a;
        if (temp < t_max && temp > t_min){ t_hit = temp; hit_p = r.at(t_hit); normal = (hit_p - s.center) / s.radius; return true; }
        temp = (-b + sq) / a;
        if (temp < t_max && temp > t_min){ t_hit = temp; hit_p = r.at(t_hit); normal = (hit_p - s.center) / s.radius; return true; }
    }
    return false;
}

Vec3 shade_simple_cpu(const Ray& r, const std::vector<Sphere>& spheres, const std::vector<Material>& mats){
    float t_hit = 1e20f; int hit_i = -1; Vec3 hit_p, normal;
    for (size_t i=0;i<spheres.size();++i){
        float ttmp; Vec3 ptmp, ntemp;
        if (hit_sphere_cpu(spheres[i], r, 0.001f, t_hit, ttmp, ptmp, ntemp)){
            t_hit = ttmp; hit_p = ptmp; normal = ntemp; hit_i = (int)i;
        }
    }
    if (hit_i==-1){
        Vec3 unit = r.d.normalized();
        float t = 0.5f*(unit.y + 1.0f);
        return Vec3(0.6f,0.8f,1.0f)*t; // sky
    }
    Material m = mats[spheres[hit_i].mat];
    if (m.type==EMISSIVE) return m.emission;
    // lambert + simple point-light at (5,5,5)
    Vec3 lightpos(5,5,5);
    Vec3 tolight = (lightpos - hit_p).normalized();
    float diff = fmaxf(0.0f, normal.dot(tolight));
    return m.albedo * diff;
}

// GPU device analogs
__device__ bool hit_sphere_device(const Sphere& s, const Ray& r, float t_min, float t_max, float &t_hit, Vec3 &hit_p, Vec3 &normal){
    Vec3 oc = r.o - s.center;
    float a = r.d.dot(r.d);
    float b = oc.dot(r.d);
    float c = oc.dot(oc) - s.radius*s.radius;
    float disc = b*b - a*c;
    if (disc > 0){
        float sq = sqrtf(disc);
        float temp = (-b - sq) / a;
        if (temp < t_max && temp > t_min){ t_hit = temp; hit_p = r.at(t_hit); normal = (hit_p - s.center) / s.radius; return true; }
        temp = (-b + sq) / a;
        if (temp < t_max && temp > t_min){ t_hit = temp; hit_p = r.at(t_hit); normal = (hit_p - s.center) / s.radius; return true; }
    }
    return false;
}

__device__ Vec3 shade_simple_device(const Ray& r, Sphere* spheres, int nspheres, Material* mats){
    float t_hit = 1e20f; int hit_i = -1; Vec3 hit_p, normal;
    for (int i=0;i<nspheres;i++){
        float ttmp; Vec3 ptmp, ntemp;
        if (hit_sphere_device(spheres[i], r, 0.001f, t_hit, ttmp, ptmp, ntemp)){
            t_hit = ttmp; hit_p = ptmp; normal = ntemp; hit_i = i;
        }
    }
    if (hit_i==-1){
        Vec3 unit = r.d.normalized();
        float t = 0.5f*(unit.y + 1.0f);
        return Vec3(0.6f,0.8f,1.0f)*t;
    }
    Material m = mats[spheres[hit_i].mat];
    if (m.type==EMISSIVE) return m.emission;
    Vec3 lightpos(5,5,5);
    Vec3 tolight = (lightpos - hit_p).normalized();
    float diff = fmaxf(0.0f, normal.dot(tolight));
    return m.albedo * diff;
}

__global__ void render_simple_gpu(Vec3* fb, int width, int height, Vec3 cam_pos, Vec3 cam_look, Sphere* d_spheres, int nspheres, Material* d_mats){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x>=width || y>=height) return;
    int idx = y*width + x;
    float aspect = float(width)/float(height);
    float u = ( (x + 0.5f) / width - 0.5f ) * 2.0f * aspect;
    float v = ( (y + 0.5f) / height - 0.5f ) * 2.0f;
    Vec3 forward = (cam_look - cam_pos).normalized();
    Vec3 right = cross(forward, Vec3(0,1,0)).normalized();
    Vec3 up = cross(right, forward).normalized();
    Vec3 dir = (forward + right * u + up * v).normalized();
    Ray r(cam_pos, dir);
    fb[idx] = shade_simple_device(r, d_spheres, nspheres, d_mats);
}

// -------------------------- Path Tracer (GPU with curand) --------------------------

__device__ float rnd_curand(curandState* st){ return curand_uniform(st); }

__device__ Vec3 random_in_unit_sphere_device(curandState* st){
    while(true){
        float x = rnd_curand(st)*2.0f - 1.0f;
        float y = rnd_curand(st)*2.0f - 1.0f;
        float z = rnd_curand(st)*2.0f - 1.0f;
        Vec3 p(x,y,z);
        if (p.dot(p) < 1.0f) return p;
    }
}

__device__ Vec3 reflect_device(const Vec3& v, const Vec3& n){ return v - n*(2.0f * v.dot(n)); }

// CPU reflect function
Vec3 reflect_cpu(const Vec3& v, const Vec3& n){ return v - n*(2.0f * v.dot(n)); }

// CPU path tracer
Vec3 trace_path_cpu(Ray r, const std::vector<Sphere>& spheres, const std::vector<Material>& mats, std::mt19937& rng, int maxDepth){
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    Vec3 throughput(1,1,1);
    Vec3 accum(0,0,0);
    for (int depth=0; depth<maxDepth; ++depth){
        float t_hit = 1e20f; int hit_i = -1; Vec3 hit_p, normal;
        for (size_t i=0; i<spheres.size(); ++i){
            float ttmp; Vec3 ptmp, ntemp;
            if (hit_sphere_cpu(spheres[i], r, 0.001f, t_hit, ttmp, ptmp, ntemp)){
                t_hit = ttmp; hit_p = ptmp; normal = ntemp; hit_i = (int)i;
            }
        }
        if (hit_i==-1){
            Vec3 unit = r.d.normalized();
            float t = 0.5f*(unit.y + 1.0f);
            Vec3 bg = Vec3(0.6f,0.8f,1.0f)*t;
            accum = accum + throughput * bg;
            break;
        }
        Material mat = mats[spheres[hit_i].mat];
        if (mat.type == EMISSIVE){ accum = accum + throughput * mat.emission; break; }
        if (mat.type == DIFFUSE){
            float x = U(rng)*2.0f - 1.0f, y = U(rng)*2.0f - 1.0f, z = U(rng)*2.0f - 1.0f;
            Vec3 p(x,y,z);
            while (p.dot(p) >= 1.0f){ x = U(rng)*2.0f - 1.0f; y = U(rng)*2.0f - 1.0f; z = U(rng)*2.0f - 1.0f; p = Vec3(x,y,z); }
            Vec3 target = hit_p + normal + p;
            r = Ray(hit_p, (target - hit_p).normalized());
            throughput = throughput * mat.albedo;
            continue;
        } else if (mat.type == METAL){
            Vec3 reflected = reflect_cpu(r.d.normalized(), normal);
            float x = U(rng)*2.0f - 1.0f, y = U(rng)*2.0f - 1.0f, z = U(rng)*2.0f - 1.0f;
            Vec3 p(x,y,z);
            while (p.dot(p) >= 1.0f){ x = U(rng)*2.0f - 1.0f; y = U(rng)*2.0f - 1.0f; z = U(rng)*2.0f - 1.0f; p = Vec3(x,y,z); }
            r = Ray(hit_p, (reflected + p * mat.fuzz).normalized());
            throughput = throughput * mat.albedo;
            if (r.d.dot(normal) <= 0) break;
            continue;
        } else {
            break;
        }
    }
    return accum;
}

__device__ bool hit_any_sphere(Sphere* spheres, int nspheres, const Ray& r, float t_min, float t_max, float& t_hit, Vec3& hit_p, Vec3& normal, int& hit_idx){
    t_hit = t_max; hit_idx = -1;
    for (int i=0;i<nspheres;i++){
        float ttmp; Vec3 ptmp, ntemp;
        if (hit_sphere_device(spheres[i], r, t_min, t_hit, ttmp, ptmp, ntemp)){
            t_hit = ttmp; hit_p = ptmp; normal = ntemp; hit_idx = i;
        }
    }
    return hit_idx != -1;
}

__device__ Vec3 trace_path_device(Ray r, Sphere* spheres, int nspheres, Material* mats, curandState* st, int maxDepth){
    Vec3 throughput(1,1,1);
    Vec3 accum(0,0,0);
    for (int depth=0; depth<maxDepth; ++depth){
        float t_hit; Vec3 hit_p, normal; int hit_idx;
        if (!hit_any_sphere(spheres, nspheres, r, 0.001f, 1e20f, t_hit, hit_p, normal, hit_idx)){
            Vec3 unit = r.d.normalized();
            float t = 0.5f*(unit.y + 1.0f);
            Vec3 bg = Vec3(0.6f,0.8f,1.0f)*t;
            accum = accum + throughput * bg;
            break;
        }
        Material mat = mats[spheres[hit_idx].mat];
        if (mat.type == EMISSIVE){ accum = accum + throughput * mat.emission; break; }
        if (mat.type == DIFFUSE){
            Vec3 target = hit_p + normal + random_in_unit_sphere_device(st);
            r = Ray(hit_p, (target - hit_p).normalized());
            throughput = throughput * mat.albedo;
            continue;
        } else if (mat.type == METAL){
            Vec3 reflected = reflect_device(r.d.normalized(), normal);
            r = Ray(hit_p, (reflected + random_in_unit_sphere_device(st) * mat.fuzz).normalized());
            throughput = throughput * mat.albedo;
            if (r.d.dot(normal) <= 0) break;
            continue;
        } else {
            break;
        }
    }
    return accum;
}

__global__ void init_curand(curandState* states, int width, int height, unsigned long seed){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x>=width || y>=height) return;
    int idx = y*width + x;
    curand_init(seed + idx, 0, 0, &states[idx]);
}

__global__ void render_path_gpu(Vec3* fb, int width, int height, Sphere* spheres, int nspheres, Material* mats, curandState* states, int samples, int maxDepth, Vec3 cam_pos, Vec3 cam_look){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x>=width || y>=height) return;
    int idx = y*width + x;
    curandState local = states[idx];
    Vec3 col(0,0,0);

    Vec3 forward = (cam_look - cam_pos).normalized();
    Vec3 right = cross(forward, Vec3(0,1,0)).normalized();
    Vec3 up = cross(right, forward).normalized();
    float fov = 60.0f; float aspect = float(width)/float(height);
    float scale = tanf(fov * 0.5f * 3.14159265f / 180.0f);

    for (int s=0; s<samples; ++s){
        float rx = rnd_curand(&local);
        float ry = rnd_curand(&local);
        float u = ( (x + rx) / float(width) - 0.5f ) * 2.0f * aspect * scale;
        float v = ( (y + ry) / float(height) - 0.5f ) * 2.0f * scale;
        Vec3 dir = (forward + right*u + up*v).normalized();
        Ray r(cam_pos, dir);
        col = col + trace_path_device(r, spheres, nspheres, mats, &local, maxDepth);
    }
    states[idx] = local;
    col = col / float(samples);
    // gamma
    fb[idx] = Vec3(sqrtf(col.x), sqrtf(col.y), sqrtf(col.z));
}

// -------------------------- Host orchestration & menu --------------------------
int main(){
    std::cout << "Render All - Ray / Path tracer\n";
    std::cout << "1 - Simple Ray Tracer (CPU/GPU/Both)\n";
    std::cout << "2 - Path Tracer (GPU) [+ optional CPU slow mode]\n";
    std::cout << "Choose: ";
    int mode; std::cin >> mode;

    // common scene setup
    std::vector<Sphere> spheres;
    std::vector<Material> mats;

    Material white; white.albedo = Vec3(0.8f,0.8f,0.8f); white.fuzz=0; white.type=DIFFUSE; white.emission = Vec3(0,0,0);
    Material red; red.albedo = Vec3(0.8f,0.1f,0.1f); red.fuzz=0; red.type=DIFFUSE; red.emission = Vec3(0,0,0);
    Material metal; metal.albedo = Vec3(0.9f,0.85f,0.8f); metal.fuzz=0.02f; metal.type=METAL; metal.emission=Vec3(0,0,0);
    Material light; light.albedo = Vec3(1,1,1); light.fuzz=0; light.type=EMISSIVE; light.emission = Vec3(12,12,12);

    mats.push_back(white); //0
    mats.push_back(red);   //1
    mats.push_back(metal); //2
    mats.push_back(light); //3

    Sphere ground; ground.center = Vec3(0,-1000.5f,-1); ground.radius=1000.0f; ground.mat = 0;
    Sphere s1; s1.center = Vec3(0,0.5f,-1); s1.radius=0.5f; s1.mat=1;
    Sphere s2; s2.center = Vec3(-1.0f,0.5f,-1.5f); s2.radius=0.5f; s2.mat=2;
    Sphere s3; s3.center = Vec3(1.0f,0.5f,-0.5f); s3.radius=0.5f; s3.mat=0;
    Sphere light_s; light_s.center = Vec3(0,5.0f,-1.0f); light_s.radius=1.0f; light_s.mat=3;

    spheres.push_back(ground);
    spheres.push_back(s1);
    spheres.push_back(s2);
    spheres.push_back(s3);
    spheres.push_back(light_s);

    // camera
    Vec3 cam_pos(0,1.0f,3.0f), cam_look(0,0.5f,-1.0f);

    if (mode==1){
        std::cout << "Simple Ray Tracer selected.\n";
        std::cout << "Choose implementation: 1-CPU, 2-GPU, 3-Both: ";
        int opt; std::cin >> opt;
        int width = 1280, height = 720;
        std::cout << "Resolution? 1-720p(1280x720) 2-1080p(1920x1080) 3-1440p(2560x1440): ";
        int r; std::cin >> r;
        if (r==2){ width=1920; height=1080; } else if (r==3){ width=2560; height=1440; }

        size_t npixels = (size_t)width * height;
        Vec3* fb = new Vec3[npixels];

        if (opt==1 || opt==3){
            auto t0 = std::chrono::high_resolution_clock::now();
            // CPU render simple
            for (int y=0;y<height;y++){
                for (int x=0;x<width;x++){
                    float aspect = float(width)/float(height);
                    float u = ( (x + 0.5f) / width - 0.5f ) * 2.0f * aspect;
                    float v = ( (y + 0.5f) / height - 0.5f ) * 2.0f;
                    Vec3 forward = (cam_look - cam_pos).normalized();
                    Vec3 right = cross(forward, Vec3(0,1,0)).normalized();
                    Vec3 up = cross(right, forward).normalized();
                    Vec3 dir = (forward + right * u + up * v).normalized();
                    Ray rcam(cam_pos, dir);
                    Vec3 col = shade_simple_cpu(rcam, spheres, mats);
                    fb[y*width + x] = col;
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> dt = t1 - t0;
            std::cout << "CPU simple render time: " << dt.count() << " s\n";
            // save CPU image
            unsigned char* img = new unsigned char[npixels*3];
            for (size_t i=0;i<npixels;i++){
                Vec3 c = fb[i];
                int ir = int(fminf(fmaxf(c.x,0.0f),1.0f) * 255.99f);
                int ig = int(fminf(fmaxf(c.y,0.0f),1.0f) * 255.99f);
                int ib = int(fminf(fmaxf(c.z,0.0f),1.0f) * 255.99f);
                img[3*i+0] = (unsigned char)ir;
                img[3*i+1] = (unsigned char)ig;
                img[3*i+2] = (unsigned char)ib;
            }
            stbi_write_png("simple_cpu.png", width, height, 3, img, width*3);
            std::cout << "Saved simple_cpu.png\n";
            delete[] img;
        }

        if (opt==2 || opt==3){
            // prepare device memory
            Sphere* d_spheres; Material* d_mats; Vec3* d_fb;
            checkCuda(cudaMalloc(&d_spheres, sizeof(Sphere) * spheres.size()), "alloc spheres");
            checkCuda(cudaMalloc(&d_mats, sizeof(Material) * mats.size()), "alloc mats");
            checkCuda(cudaMalloc(&d_fb, sizeof(Vec3) * npixels), "alloc fb");

            checkCuda(cudaMemcpy(d_spheres, spheres.data(), sizeof(Sphere) * spheres.size(), cudaMemcpyHostToDevice), "copy spheres");
            checkCuda(cudaMemcpy(d_mats, mats.data(), sizeof(Material) * mats.size(), cudaMemcpyHostToDevice), "copy mats");

            dim3 threads(16,16);
            dim3 blocks( (width + threads.x -1)/threads.x, (height + threads.y -1)/threads.y );

            auto t0 = std::chrono::high_resolution_clock::now();
            render_simple_gpu<<<blocks, threads>>>(d_fb, width, height, cam_pos, cam_look, d_spheres, (int)spheres.size(), d_mats);
            checkCuda(cudaGetLastError(), "launch simple kernel");
            checkCuda(cudaDeviceSynchronize(), "sync after simple kernel");
            auto t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> dt = t1 - t0;
            std::cout << "GPU simple render kernel time: " << dt.count() << " s\n";

            // copy back
            checkCuda(cudaMemcpy(fb, d_fb, sizeof(Vec3) * npixels, cudaMemcpyDeviceToHost), "copy fb");
            // save
            unsigned char* img = new unsigned char[npixels*3];
            for (size_t i=0;i<npixels;i++){
                Vec3 c = fb[i];
                int ir = int(fminf(fmaxf(c.x,0.0f),1.0f) * 255.99f);
                int ig = int(fminf(fmaxf(c.y,0.0f),1.0f) * 255.99f);
                int ib = int(fminf(fmaxf(c.z,0.0f),1.0f) * 255.99f);
                img[3*i+0] = (unsigned char)ir;
                img[3*i+1] = (unsigned char)ig;
                img[3*i+2] = (unsigned char)ib;
            }
            stbi_write_png("simple_gpu.png", width, height, 3, img, width*3);
            std::cout << "Saved simple_gpu.png\n";
            delete[] img;

            cudaFree(d_spheres); cudaFree(d_mats); cudaFree(d_fb);
        }

        delete[] fb;
    }
    else if (mode==2){
        std::cout << "Path Tracer selected (GPU). This can be slow with high samples/res.\n";
        std::cout << "Enter width (e.g. 1280) and height (e.g. 720):\n";
        int width=1280, height=720;
        std::cin >> width >> height;
        std::cout << "Samples per pixel (e.g. 50 or 200): "; int samples; std::cin >> samples;
        std::cout << "Max bounces (depth) (e.g. 4 or 6): "; int maxDepth; std::cin >> maxDepth;
        std::cout << "Choose CPU implementation: 0-none 1-single-thread 2-OpenMP parallel 3-both: "; int runCPU; std::cin >> runCPU;

        size_t npixels = (size_t)width * height;
        Vec3* fb = new Vec3[npixels];

        // prepare device data
        Sphere* d_spheres; Material* d_mats; Vec3* d_fb; curandState* d_states;
        checkCuda(cudaMalloc(&d_spheres, sizeof(Sphere) * spheres.size()), "alloc spheres");
        checkCuda(cudaMalloc(&d_mats, sizeof(Material) * mats.size()), "alloc mats");
        checkCuda(cudaMalloc(&d_fb, sizeof(Vec3) * npixels), "alloc fb");
        checkCuda(cudaMalloc(&d_states, sizeof(curandState) * npixels), "alloc states");
        checkCuda(cudaMemcpy(d_spheres, spheres.data(), sizeof(Sphere) * spheres.size(), cudaMemcpyHostToDevice), "copy spheres");
        checkCuda(cudaMemcpy(d_mats, mats.data(), sizeof(Material) * mats.size(), cudaMemcpyHostToDevice), "copy mats");

        dim3 threads(16,16);
        dim3 blocks( (width + threads.x -1)/threads.x, (height + threads.y -1)/threads.y );

        init_curand<<<blocks, threads>>>(d_states, width, height, (unsigned long)time(NULL));
        checkCuda(cudaGetLastError(), "init_curand");
        checkCuda(cudaDeviceSynchronize(), "sync after init_curand");

        auto t0 = std::chrono::high_resolution_clock::now();
        render_path_gpu<<<blocks, threads>>>(d_fb, width, height, d_spheres, (int)spheres.size(), d_mats, d_states, samples, maxDepth, cam_pos, cam_look);
        checkCuda(cudaGetLastError(), "render_path_gpu");
        checkCuda(cudaDeviceSynchronize(), "sync after render_path");
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt = t1 - t0;
        std::cout << "GPU path render time: " << dt.count() << " s\n";

        checkCuda(cudaMemcpy(fb, d_fb, sizeof(Vec3) * npixels, cudaMemcpyDeviceToHost), "copy fb");

        // save
        unsigned char* img = new unsigned char[npixels*3];
        for (size_t i=0;i<npixels;i++){
            Vec3 c = fb[i];
            int ir = int(fminf(fmaxf(c.x,0.0f),1.0f) * 255.99f);
            int ig = int(fminf(fmaxf(c.y,0.0f),1.0f) * 255.99f);
            int ib = int(fminf(fmaxf(c.z,0.0f),1.0f) * 255.99f);
            img[3*i+0] = (unsigned char)ir;
            img[3*i+1] = (unsigned char)ig;
            img[3*i+2] = (unsigned char)ib;
        }
        stbi_write_png("path_gpu.png", width, height, 3, img, width*3);
        std::cout << "Saved path_gpu.png\n";
        delete[] img;

        // CPU reference options
        if (runCPU==1 || runCPU==3){
            std::cout << "Running single-thread CPU reference... this may take a long time.\n";
            std::mt19937 rng(12345);
            std::uniform_real_distribution<float> U(0.0f,1.0f);
            auto t0c = std::chrono::high_resolution_clock::now();
            for (int y=0;y<height;y++){
                for (int x=0;x<width;x++){
                    Vec3 col(0,0,0);
                    for (int s=0;s<samples;s++){
                        float rx = U(rng), ry = U(rng);
                        float u = ( (x + rx) / float(width) - 0.5f ) * 2.0f * (float(width)/float(height)) * tanf(60.0f*0.5f*3.14159265f/180.0f);
                        float v = ( (y + ry) / float(height) - 0.5f ) * 2.0f * tanf(60.0f*0.5f*3.14159265f/180.0f);
                        Vec3 forward = (cam_look - cam_pos).normalized();
                        Vec3 right = cross(forward, Vec3(0,1,0)).normalized();
                        Vec3 up = cross(right, forward).normalized();
                        Vec3 dir = (forward + right*u + up*v).normalized();
                        Ray r(cam_pos, dir);
                        col = col + trace_path_cpu(r, spheres, mats, rng, maxDepth);
                    }
                    col = col / float(samples);
                    fb[y*width + x] = Vec3(sqrtf(col.x), sqrtf(col.y), sqrtf(col.z));
                }
            }
            auto t1c = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> dtc = t1c - t0c;
            std::cout << "CPU path (single-thread) time: " << dtc.count() << " s\n";
            // save
            unsigned char* imgc = new unsigned char[npixels*3];
            for (size_t i=0;i<npixels;i++){
                Vec3 c = fb[i];
                int ir = int(fminf(fmaxf(c.x,0.0f),1.0f) * 255.99f);
                int ig = int(fminf(fmaxf(c.y,0.0f),1.0f) * 255.99f);
                int ib = int(fminf(fmaxf(c.z,0.0f),1.0f) * 255.99f);
                imgc[3*i+0] = (unsigned char)ir;
                imgc[3*i+1] = (unsigned char)ig;
                imgc[3*i+2] = (unsigned char)ib;
            }
            stbi_write_png("path_cpu_ref.png", width, height, 3, imgc, width*3);
            std::cout << "Saved path_cpu_ref.png\n";
            delete[] imgc;
        }
        if (runCPU==2 || runCPU==3){
            std::cout << "Running OpenMP parallel CPU Path Tracer.\n";
            std::cout << "How many threads? (0=auto, or specify number): ";
            int nthreads;
            std::cin >> nthreads;
            if (nthreads <= 0) nthreads = omp_get_max_threads();
            omp_set_num_threads(nthreads);
            std::cout << "Using " << nthreads << " threads.\n";

            auto t0c = std::chrono::high_resolution_clock::now();

            #pragma omp parallel for collapse(2) schedule(dynamic, 1)
            for (int y=0; y<height; y++){
                for (int x=0; x<width; x++){
                    // Each thread gets its own RNG
                    std::mt19937 local_rng(12345 + y * width + x);
                    std::uniform_real_distribution<float> U(0.0f, 1.0f);
                    Vec3 col(0,0,0);
                    
                    for (int s=0; s<samples; s++){
                        float rx = U(local_rng), ry = U(local_rng);
                        float u = ( (x + rx) / float(width) - 0.5f ) * 2.0f * (float(width)/float(height)) * tanf(60.0f*0.5f*3.14159265f/180.0f);
                        float v = ( (y + ry) / float(height) - 0.5f ) * 2.0f * tanf(60.0f*0.5f*3.14159265f/180.0f);
                        Vec3 forward = (cam_look - cam_pos).normalized();
                        Vec3 right = cross(forward, Vec3(0,1,0)).normalized();
                        Vec3 up = cross(right, forward).normalized();
                        Vec3 dir = (forward + right*u + up*v).normalized();
                        Ray r(cam_pos, dir);
                        col = col + trace_path_cpu(r, spheres, mats, local_rng, maxDepth);
                    }
                    col = col / float(samples);
                    fb[y*width + x] = Vec3(sqrtf(col.x), sqrtf(col.y), sqrtf(col.z));
                }
            }

            auto t1c = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> dtc = t1c - t0c;
            std::cout << "CPU path (OpenMP " << nthreads << " threads) time: " << dtc.count() << " s\n";
            
            // save
            unsigned char* imgc = new unsigned char[npixels*3];
            for (size_t i=0;i<npixels;i++){
                Vec3 c = fb[i];
                int ir = int(fminf(fmaxf(c.x,0.0f),1.0f) * 255.99f);
                int ig = int(fminf(fmaxf(c.y,0.0f),1.0f) * 255.99f);
                int ib = int(fminf(fmaxf(c.z,0.0f),1.0f) * 255.99f);
                imgc[3*i+0] = (unsigned char)ir;
                imgc[3*i+1] = (unsigned char)ig;
                imgc[3*i+2] = (unsigned char)ib;
            }
            stbi_write_png("path_cpu_omp.png", width, height, 3, imgc, width*3);
            std::cout << "Saved path_cpu_omp.png\n";
            delete[] imgc;
        }

        // free device
        cudaFree(d_spheres); cudaFree(d_mats); cudaFree(d_fb); cudaFree(d_states);
        delete[] fb;
    }
    else {
        std::cout << "Invalid mode\n";
    }

    std::cout << "Done.\n";
    return 0;
}
