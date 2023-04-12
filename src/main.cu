#include <iostream>
#include <string>
#include <cfloat>
#include <curand_kernel.h>
#include "base/image.h"
#include "shapes/sphere.h"
#include "base/world.h"
#include "base/camera.h"
#include "base/material.h"

using namespace std;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '"
             << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(Shape **gpu_shape_list, World **gpu_world, Camera **gpu_camera, uint width,
                             uint height, curandState *rand_state) {
    curandState local_rand_state = *rand_state;
    gpu_shape_list[0] = new Sphere(Point(0, -1000.0, -1), 1000, new Lambertian(Color(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = RND;
            Point center(a + RND, 0.2, b + RND);
            if (choose_mat < 0.8f) {
                gpu_shape_list[i++] =
                    new Sphere(center, 0.2, new Lambertian(Color(RND * RND, RND * RND, RND * RND)));
            } else if (choose_mat < 0.95f) {
                gpu_shape_list[i++] =
                    new Sphere(center, 0.2,
                               new Metal(Color(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)),
                                         0.5f * RND));
            } else {
                gpu_shape_list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
            }
        }
    }

    gpu_shape_list[i++] = new Sphere(Point(0, 1, 0), 1.0, new Dielectric(1.5));
    gpu_shape_list[i++] = new Sphere(Point(-4, 1, 0), 1.0, new Lambertian(Color(0.4, 0.2, 0.1)));
    gpu_shape_list[i++] = new Sphere(Point(4, 1, 0), 1.0, new Metal(Color(0.7, 0.6, 0.5), 0.0));
    *rand_state = local_rand_state;
    *gpu_world = new World(gpu_shape_list, 22 * 22 + 1 + 3);

    Point look_from(13, 2, 3);
    Point look_at(0, 0, 0);
    float dist_to_focus = (look_from - look_at).length();
    float aperture = 0.1;
    *gpu_camera = new Camera(look_from, look_at, Vector3(0, 1, 0), 30.0, float(width) / float(height),
                             aperture, dist_to_focus);
}

__global__ void free_world(World **gpu_world, Camera **gpu_camera) {
    for (int idx = 0; idx < (*gpu_world)->size; idx++) {
        delete (*gpu_world)->list[idx]->get_material_ptr();
        delete (*gpu_world)->list[idx];
    }
    delete *gpu_world;
    delete *gpu_camera;
}

__device__ Color ray_trace(const Ray &ray, World **world, curandState *local_rand_state) {
    Ray current_ray = ray;
    Color current_attenuation = Color(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        Intersection intersection;
        if ((*world)->intersect(intersection, current_ray, 0.001f, FLT_MAX)) {
            Ray scattered_ray;
            Color attenuation;
            if (!intersection.material_ptr->scatter(current_ray, intersection, attenuation, scattered_ray,
                                                    local_rand_state)) {
                return Color(0.0, 0.0, 0.0);
            }

            current_attenuation *= attenuation;
            current_ray = scattered_ray;
            continue;
        }

        float t = 0.5f * (current_ray.d.normalize().y + 1.0f);
        Vector3 c = (1.0f - t) * Vector3(1.0, 1.0, 1.0) + t * Vector3(0.5, 0.7, 1.0);
        return current_attenuation * Color(c.x, c.y, c.z);
    }

    return Color(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    curand_init(1984, 0, 0, rand_state);
}

__global__ void render_init(uint width, uint height, curandState *rand_state) {
    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height))
        return;
    uint pixel_index = y * width + x;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(Color *frame_buffer, uint width, uint height, uint num_samples, Camera **camera,
                       World **world, curandState *rand_state) {
    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) {
        return;
    }

    int pixel_index = y * width + x;
    curandState local_rand_state = rand_state[pixel_index];

    Color final_color(0, 0, 0);
    for (uint s = 0; s < num_samples; s++) {
        float u = float(x + curand_uniform(&local_rand_state)) / float(width);
        float v = float(y + curand_uniform(&local_rand_state)) / float(height);
        final_color += ray_trace((*camera)->get_ray(u, v, &local_rand_state), world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    final_color /= float(num_samples);

    final_color = Color(sqrt(final_color.r), sqrt(final_color.g), sqrt(final_color.b));
    frame_buffer[y * width + x] = final_color;
}

void writer_to_file(const string &file_name, uint width, uint height, const Color *frame_buffer) {
    Image image(frame_buffer, width, height);
    image.flip();
    image.writePNG(file_name);
}

int main() {
    uint width = 1960;
    uint height = 1080;
    uint thread_width = 8;
    uint thread_height = 8;
    uint num_samples = 100;

    cerr << "Rendering a " << width << "x" << height << " image (samples per pixel: " << num_samples << ") ";
    cerr << "in " << thread_width << "x" << thread_height << " blocks.\n";

    // allocate FB
    Color *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, sizeof(Color) * width * height));

    // allocate random state
    curandState *gpu_rand_state;
    checkCudaErrors(cudaMalloc((void **)&gpu_rand_state, width * height * sizeof(curandState)));

    curandState *gpu_rand_create_world;
    checkCudaErrors(cudaMalloc((void **)&gpu_rand_create_world, sizeof(curandState)));

    rand_init<<<1, 1>>>(gpu_rand_create_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Shape **gpu_shape_list;
    uint num_sphere = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void **)&gpu_shape_list, num_sphere * sizeof(Shape *)));
    World **gpu_world;
    checkCudaErrors(cudaMalloc((void **)&gpu_world, sizeof(World *)));
    Camera **gpu_camera;
    checkCudaErrors(cudaMalloc((void **)&gpu_camera, sizeof(Camera *)));

    create_world<<<1, 1>>>(gpu_shape_list, gpu_world, gpu_camera, width, height, gpu_rand_create_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start = clock();
    dim3 blocks(width / thread_width + 1, height / thread_height + 1, 1);
    dim3 threads(thread_width, thread_height, 1);

    render_init<<<blocks, threads>>>(width, height, gpu_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(frame_buffer, width, height, num_samples, gpu_camera, gpu_world,
                                gpu_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    double timer_seconds = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    cerr << "took " << timer_seconds << " seconds.\n";

    string file_name = "output.png";
    writer_to_file(file_name, width, height, frame_buffer);

    free_world<<<1, 1>>>(gpu_world, gpu_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(gpu_shape_list));
    checkCudaErrors(cudaFree(gpu_world));
    checkCudaErrors(cudaFree(gpu_camera));
    checkCudaErrors(cudaFree(gpu_rand_state));
    checkCudaErrors(cudaFree(gpu_rand_create_world));
    checkCudaErrors(cudaFree(frame_buffer));
    cudaDeviceReset();

    cout << "image saved to `" << file_name << "`\n";

    return 0;
}
