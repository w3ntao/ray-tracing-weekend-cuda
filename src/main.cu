#include <iostream>
#include <string>
#include "base/image.cuh"
#include "base/vector3.cuh"
#include "base/point3.cuh"
#include "base/ray.cuh"

using namespace std;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line
                  << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

//__global__ void render(RGBColor *o_frame_buffer, int width, int height) {
__global__ void render(Color *frame_buffer, int width, int height) {
    int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((pixel_x >= width) || (pixel_y >= height)) {
        return;
    }
    int pixel_index = pixel_y * width + pixel_x;
    frame_buffer[pixel_index] = Color(float(pixel_x) / width, float(pixel_y) / height, 0.2);
}

void writer_to_file(const string &file_name, int width, int height, const Color *frame_buffer) {
    Image image(frame_buffer, width, height);
    image.writePNG(file_name);
}

int main() {
    int width = 1960;
    int height = 1080;
    int thread_width = 8;
    int thread_height = 8;

    std::cerr << "Rendering a " << width << "x" << height << " image ";
    std::cerr << "in " << thread_width << "x" << thread_height << " blocks.\n";

    // allocate FB
    Color *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, 3 * sizeof(Color) * width * height));

    clock_t start = clock();
    // Render our buffer
    dim3 blocks(width / thread_width + 1, height / thread_height + 1, 1);
    dim3 threads(thread_width, thread_height, 1);

    render<<<blocks, threads>>>(frame_buffer, width, height);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    double timer_seconds = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    string file_name = "output.png";

    writer_to_file(file_name, width, height, frame_buffer);

    checkCudaErrors(cudaFree(frame_buffer));

    cout << "image saved to `" << file_name << "`\n";

    return 0;
}
