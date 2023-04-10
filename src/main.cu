#include <iostream>
#include <string>
#include <cfloat>
#include "base/image.cuh"
#include "base/sphere.h"
#include "base/world.h"

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

__device__ bool hit_sphere(const Point3 &center, float radius, const Ray &r) {
    Vector3 oc = r.o - center;
    float a = dot(r.d, r.d);
    float b = 2.0f * dot(oc, r.d);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;
    return (discriminant > 0.0f);
}

__device__ Color color(const Ray &r, World **world) {
    Intersection intersection;
    if ((*world)->intersect(intersection, r, 0.0, FLT_MAX)) {
        auto result = 0.5f * Vector3(intersection.n.x + 1.0f, intersection.n.y + 1.0f,
                                     intersection.n.z + 1.0f);
        result = result.normalize();
        return Color(result.x, result.y, result.z);
    }

    Vector3 unit_direction = r.d.normalize();
    float t = 0.5f * (unit_direction.y + 1.0f);
    auto result = (1.0f - t) * Vector3(1.0, 1.0, 1.0) + t * Vector3(0.5, 0.7, 1.0);

    return Color(result.x, result.y, result.z);
}

__global__ void render(Color *frame_buffer, int width, int height, Point3 lower_left_corner,
                       Vector3 horizontal, Vector3 vertical, Point3 origin, World **world) {
    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) {
        return;
    }

    float u = float(x) / float(width);
    float v = float(y) / float(height);

    Ray ray(origin, (lower_left_corner + u * horizontal + v * vertical).to_vector());
    frame_buffer[y * width + x] = color(ray, world);
}

void writer_to_file(const string &file_name, int width, int height, const Color *frame_buffer) {
    Image image(frame_buffer, width, height);
    image.writePNG(file_name);
}

__global__ void create_world(Shape **d_list, World **d_world) {
    *(d_list) = new Sphere(Point3(0, 0, -1), 0.5);
    *(d_list + 1) = new Sphere(Point3(0, -100.5, -1), 100);
    *d_world = new World(d_list, 2);
}

__global__ void free_world(World **d_world) {
    for (int idx = 0; idx < (*d_world)->size; idx++) {
        delete (*d_world)->list[idx];
    }
    delete *d_world;
}

int main() {
    int width = 1600;
    int height = 800;
    int thread_width = 8;
    int thread_height = 8;
    std::cerr << "Rendering a " << width << "x" << height << " image ";
    std::cerr << "in " << thread_width << "x" << thread_height << " blocks.\n";

    // allocate FB
    Color *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, sizeof(Color) * width * height));

    Shape **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2 * sizeof(Shape *)));
    World **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(World *)));
    create_world<<<1, 1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start = clock();
    // Render our buffer
    dim3 blocks(width / thread_width + 1, height / thread_height + 1, 1);
    dim3 threads(thread_width, thread_height, 1);

    render<<<blocks, threads>>>(frame_buffer, width, height, Point3(-2.0, -1.0, -1.0),
                                Vector3(4.0, 0.0, 0.0), Vector3(0.0, 2.0, 0.0),
                                Point3(0.0, 0.0, 0.0), d_world);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    free_world<<<1, 1>>>(d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));

    double timer_seconds = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    string file_name = "output.png";
    writer_to_file(file_name, width, height, frame_buffer);

    checkCudaErrors(cudaFree(frame_buffer));

    cout << "image saved to `" << file_name << "`\n";

    return 0;
}
