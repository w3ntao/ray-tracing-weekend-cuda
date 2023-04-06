#include <iostream>
#include <string>
#include "png_writer.cuh"

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

__global__ void render(float *o_frame_buffer, int width, int height) {
    int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((pixel_x >= width) || (pixel_y >= height)) {
        return;
    }
    int pixel_index = pixel_y * width * 3 + pixel_x * 3;
    o_frame_buffer[pixel_index + 0] = float(pixel_x) / width;
    o_frame_buffer[pixel_index + 1] = float(pixel_y) / height;
    o_frame_buffer[pixel_index + 2] = 0.2;
}

void writer_to_file(const string &file_name, int width, int height, const float *float_buffer) {
    PngWriter png(width, height);

    float scalar = 256 - 0.0001;

    // set some pixels....
    for (int i = 0; i < width; ++i) {
        for (int k = 0; k < height; ++k) {
            size_t pixel_index = k * 3 * width + i * 3;

            int red = int(scalar * float_buffer[pixel_index + 0]);
            int green = int(scalar * float_buffer[pixel_index + 1]);
            int blue = int(scalar * float_buffer[pixel_index + 2]);

            png.set(i, k, red, green, blue); // set function assumes (0,0) is bottom left
        }
    }

    png.write(file_name);
}

int main() {
    int width = 1960;
    int height = 1080;
    int thread_width = 8;
    int thread_height = 8;

    std::cerr << "Rendering a " << width << "x" << height << " image ";
    std::cerr << "in " << thread_width << "x" << thread_height << " blocks.\n";

    // allocate FB
    float *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, 3 * sizeof(float) * width * height));

    clock_t start = clock();
    // Render our buffer
    dim3 blocks(width / thread_width + 1, height / thread_height + 1);
    dim3 threads(thread_width, thread_height);
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
