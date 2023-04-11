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

void check_cuda(cudaError_t result, char const *const func, const char *const file,
                int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":"
                  << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void create_world(Shape **gpu_shape_list, World **gpu_world, Camera **gpu_camera) {
    printf("you shouldn't invoke this function\n");

    /*
     *(gpu_shape_list + 0) = new Sphere(Point(0, 0, -1), 0.5);
     *(gpu_shape_list + 1) = new Sphere(Point(0, -100.5, -1), 100);
     *gpu_world = new World(gpu_shape_list, 2);
     *gpu_camera = new Camera();
     * */
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world_new(Shape **d_list, World **d_world, Camera **d_camera, int nx, int ny,
                                 curandState *rand_state) {
    curandState local_rand_state = *rand_state;
    d_list[0] = new Sphere(Point(0, -1000.0, -1), 1000, new lambertian(Color(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = RND;
            Point center(a + RND, 0.2, b + RND);
            if (choose_mat < 0.8f) {
                d_list[i++] =
                    new Sphere(center, 0.2, new lambertian(Color(RND * RND, RND * RND, RND * RND)));
            } else if (choose_mat < 0.95f) {
                d_list[i++] = new Sphere(
                    center, 0.2,
                    new metal(Color(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)),
                              0.5f * RND));
            } else {
                d_list[i++] = new Sphere(center, 0.2, new dielectric(1.5));
            }
        }
    }

    d_list[i++] = new Sphere(Point(0, 1, 0), 1.0, new dielectric(1.5));
    d_list[i++] = new Sphere(Point(-4, 1, 0), 1.0, new lambertian(Color(0.4, 0.2, 0.1)));
    d_list[i++] = new Sphere(Point(4, 1, 0), 1.0, new metal(Color(0.7, 0.6, 0.5), 0.0));
    *rand_state = local_rand_state;
    *d_world = new World(d_list, 22 * 22 + 1 + 3);

    Point lookfrom(13, 2, 3);
    Point lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    (lookfrom - lookat).length();
    float aperture = 0.1;
    *d_camera = new Camera(lookfrom, lookat, Vector3(0, 1, 0), 30.0, float(nx) / float(ny),
                           aperture, dist_to_focus);
}

__global__ void free_world(World **gpu_world, Camera **gpu_camera) {
    for (int idx = 0; idx < (*gpu_world)->size; idx++) {
        delete (*gpu_world)->list[idx];
    }
    delete *gpu_world;
    delete *gpu_camera;
}

__device__ Color color(const Ray &r, World **world, curandState *local_rand_state) {
    Ray cur_ray = r;
    float cur_attenuation = 1.0f;
    for (int i = 0; i < 50; i++) {
        Intersection intersection;
        if ((*world)->intersect(intersection, cur_ray, 0.001f, FLT_MAX)) {
            Point target =
                intersection.p + intersection.n + random_in_unit_sphere(local_rand_state);
            cur_attenuation *= 0.5f;
            cur_ray = Ray(intersection.p, target - intersection.p);
            continue;
        }

        Vector3 unit_direction = cur_ray.d.normalize();
        float t = 0.5f * (unit_direction.y + 1.0f);
        Vector3 c = (1.0f - t) * Vector3(1.0, 1.0, 1.0) + t * Vector3(0.5, 0.7, 1.0);

        auto result = cur_attenuation * c;
        return Color(result.x, result.y, result.z);
    }
    return Color(0.0, 0.0, 0.0); // exceeded recursion
}

__device__ Color color_new(const Ray &r, World **world, curandState *local_rand_state) {
    Ray cur_ray = r;
    Color cur_attenuation = Color(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        Intersection rec;
        if ((*world)->intersect(rec, cur_ray, 0.001f, FLT_MAX)) {
            Ray scattered;
            Color attenuation;

            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return Color(0.0, 0.0, 0.0);
            }

        } else {
            Vector3 unit_direction = cur_ray.d.normalize();
            float t = 0.5f * (unit_direction.y + 1.0f);
            Vector3 c = (1.0f - t) * Vector3(1.0, 1.0, 1.0) + t * Vector3(0.5, 0.7, 1.0);
            return cur_attenuation * Color(c.x, c.y, c.z);
        }
    }
    return Color(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    curand_init(1984, 0, 0, rand_state);
}

__global__ void render_init(int width, int height, curandState *rand_state) {
    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height))
        return;
    uint pixel_index = y * width + x;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(Color *frame_buffer, int width, int height, int num_samples, Camera **camera,
                       World **world, curandState *rand_state) {
    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) {
        return;
    }

    int pixel_index = y * width + x;
    curandState local_rand_state = rand_state[pixel_index];

    Color final_color(0, 0, 0);
    for (int s = 0; s < num_samples; s++) {
        float u = float(x + curand_uniform(&local_rand_state)) / float(width);
        float v = float(y + curand_uniform(&local_rand_state)) / float(height);
        final_color += color_new((*camera)->get_ray(u, v, &local_rand_state), world, &local_rand_state);
    }

    // final_color = Color(0.2,0.3,0.8);

    rand_state[pixel_index] = local_rand_state;
    final_color /= float(num_samples);

    final_color = Color(sqrt(final_color.r), sqrt(final_color.g), sqrt(final_color.b));
    frame_buffer[y * width + x] = final_color;
}

void writer_to_file(const string &file_name, int width, int height, const Color *frame_buffer) {
    Image image(frame_buffer, width, height);
    image.flip();
    image.writePNG(file_name);
}

int main() {
    int width = 1600;
    int height = 800;
    int thread_width = 8;
    int thread_height = 8;
    int num_samples = 10;

    std::cerr << "Rendering a " << width << "x" << height
              << " image (samples per pixel: " << num_samples << ") ";
    std::cerr << "in " << thread_width << "x" << thread_height << " blocks.\n";

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

    create_world_new<<<1, 1>>>(gpu_shape_list, gpu_world, gpu_camera, width, height,
                               gpu_rand_create_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start = clock();
    // Render our buffer
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
    std::cerr << "took " << timer_seconds << " seconds.\n";

    string file_name = "output.png";
    writer_to_file(file_name, width, height, frame_buffer);

    free_world<<<1, 1>>>(gpu_world, gpu_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(gpu_shape_list));
    checkCudaErrors(cudaFree(gpu_world));
    checkCudaErrors(cudaFree(gpu_camera));
    checkCudaErrors(cudaFree(gpu_rand_state));
    checkCudaErrors(cudaFree(frame_buffer));

    cout << "image saved to `" << file_name << "`\n";

    return 0;
}
