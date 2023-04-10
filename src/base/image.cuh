//
// Created by wentao on 4/6/23.
//

#ifndef CUDA_RAY_TRACER_IMAGE_CUH
#define CUDA_RAY_TRACER_IMAGE_CUH

#include <vector>
#include <string>
#include "util.h"
#include "color.cuh"

class Image {
    private:
        std::vector<Color> pixels;
        uint width, height;

    public:
        Image() : width(0), height(0) {}

        Image(uint _width, uint _height) : width(_width), height(_height) {
            pixels = std::vector<Color>(_width * _height);
        }

        Image(const Color *frame_buffer, uint _width, uint _height)
            : width(_width), height(_height) {
            pixels = std::vector<Color>(_width * _height);
            for (uint x = 0; x < width; ++x) {
                for (uint y = 0; y < height; ++y) {
                    size_t pixel_index = y * width + x;
                    (*this)(x, y) = frame_buffer[pixel_index];
                }
            }
        }

        Image &operator=(const Image &other) {
            width = other.width;
            height = other.height;
            pixels = other.pixels;
            return *this;
        }

        void create(uint _width, uint _height) {
            width = _width;
            height = _height;
            pixels = std::vector<Color>(_width * _height);
        }

        Color &operator()(uint x, uint y) {
            return pixels[y * width + x];
        }

        const Color &operator()(uint x, uint y) const {
            return pixels[y * width + x];
        }

        void writePNG(const std::string &file_name);

        void readPNG(const std::string &file_name);
};
#endif // CUDA_RAY_TRACER_IMAGE_CUH
