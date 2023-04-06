//
// Created by wentao on 4/6/23.
//

#ifndef CUDA_RAY_TRACER_IMAGE_CUH
#define CUDA_RAY_TRACER_IMAGE_CUH

#include <string>
#include "util.h"
#include "color.cuh"

class Image {
  public:
    Image();

    Image(uint width, uint height);

    Image(const Image &other);

    ~Image();

    void create(uint width, uint height);

    void destroy();

    RGBColor &operator()(uint x, uint y);

    const RGBColor &operator()(uint x, uint y) const;

    void clear(const RGBColor &color);

    void writePNG(const std::string &filename);

    void readPNG(const std::string &filename);

    uint width() const {
        return width_;
    }

    uint height() const {
        return height_;
    }

    Image &operator=(const Image &other);

  private:
    RGBColor *pixels;
    uint width_, height_;
};
#endif // CUDA_RAY_TRACER_IMAGE_CUH
