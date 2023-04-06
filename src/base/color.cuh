//
// Created by wentao on 4/6/23.
//

#ifndef CUDA_RAY_TRACER_COLOR_CUH
#define CUDA_RAY_TRACER_COLOR_CUH

class RGBColor {
  public:
    float r, g, b;

    __host__ __device__ RGBColor() : r(0.0), g(0.0), b(0.0) {}

    __host__ __device__ RGBColor(float r, float g, float b) : r(r), g(g), b(b) {}

    __host__ __device__ RGBColor operator+(const RGBColor &c) const {
        return RGBColor(r + c.r, g + c.g, b + c.b);
    }

    __host__ __device__ void operator+=(const RGBColor &c) {
        r += c.r;
        g += c.g;
        b += c.b;
    }

    __host__ __device__ RGBColor operator-(const RGBColor &c) const {
        return RGBColor(r - c.r, g - c.g, b - c.b);
    }

    __host__ __device__ RGBColor operator*(const RGBColor &c) const {
        return RGBColor(r * c.r, g * c.g, b * c.b);
    }

    __host__ __device__ RGBColor operator*(float scalar) const {
        return RGBColor(r * scalar, g * scalar, b * scalar);
    }

    __host__ __device__ RGBColor operator/(float divisor) const {
        return RGBColor(r / divisor, g / divisor, b / divisor);
    }

    __host__ __device__ bool operator==(const RGBColor &c) const {
        return r == c.r && g == c.g && b == c.b;
    }
    __host__ __device__ bool operator!=(const RGBColor &c) const {
        return !(*this == c);
    }

    __host__ __device__ RGBColor clamp() const {
        return RGBColor(single_clamp(r), single_clamp(g), single_clamp(b));
    }

  private:
    __host__ __device__ inline float single_clamp(float x) const {
        return x < 0 ? 0 : (x > 1 ? 1 : x);
    }
};

#endif // CUDA_RAY_TRACER_COLOR_CUH
