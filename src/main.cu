#include <iostream>
#include <string>
#include <cfloat>
#include <curand_kernel.h>
#include "base/image.h"
#include "base/sphere.h"
#include "base/world.h"
#include "base/camera.h"

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
    *(gpu_shape_list + 0) = new Sphere(Point3(0, 0, -1), 0.5);
    *(gpu_shape_list + 1) = new Sphere(Point3(0, -100.5, -1), 100);
    *gpu_world = new World(gpu_shape_list, 2);
    *gpu_camera = new Camera();
}

__global__ void free_world(World **gpu_world, Camera **gpu_camera) {
    for (int idx = 0; idx < (*gpu_world)->size; idx++) {
        delete (*gpu_world)->list[idx];
    }
    delete *gpu_world;
    delete *gpu_camera;
}

__device__ Vector3 random_in_unit_sphere(curandState *local_rand_state) {
    Vector3 p;
    do {
        auto random_vector =
            Vector3(curand_uniform(local_rand_state), curand_uniform(local_rand_state),
                    curand_uniform(local_rand_state));
        p = 2.0f * random_vector - Vector3(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ Color color(const Ray &r, World **world, curandState *local_rand_state) {
    Ray cur_ray = r;
    float cur_attenuation = 1.0f;
    for (int i = 0; i < 50; i++) {
        Intersection intersection;
        if ((*world)->intersect(intersection, cur_ray, 0.001f, FLT_MAX)) {
            Point3 target =
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

__global__ void render_init(int width, int height, curandState *rand_state) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height))
        return;
    int pixel_index = y * width + x;
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
        final_color += color((*camera)->get_ray(u, v), world, &local_rand_state);
    }

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
    int num_samples = 100;

    std::cerr << "Rendering a " << width << "x" << height
              << " image (samples per pixel: " << num_samples << ") ";
    std::cerr << "in " << thread_width << "x" << thread_height << " blocks.\n";

    // allocate random state
    curandState *gpu_rand_state;
    checkCudaErrors(cudaMalloc((void **)&gpu_rand_state, width * height * sizeof(curandState)));

    // allocate FB
    Color *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, sizeof(Color) * width * height));

    Shape **gpu_shape_list;
    checkCudaErrors(cudaMalloc((void **)&gpu_shape_list, 2 * sizeof(Shape *)));
    World **gpu_world;
    checkCudaErrors(cudaMalloc((void **)&gpu_world, sizeof(World *)));
    Camera **gpu_camera;
    checkCudaErrors(cudaMalloc((void **)&gpu_camera, sizeof(Camera *)));
    create_world<<<1, 1>>>(gpu_shape_list, gpu_world, gpu_camera);
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
