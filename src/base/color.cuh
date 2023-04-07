//
// Created by wentao on 4/6/23.
//

#ifndef CUDA_RAY_TRACER_COLOR_CUH
#define CUDA_RAY_TRACER_COLOR_CUH

class Color {
  public:
    float r, g, b;

    __host__ __device__ Color() : r(0.0), g(0.0), b(0.0) {}

    __host__ __device__ Color(float r, float g, float b) : r(r), g(g), b(b) {}

    __device__ Color operator+(const Color &c) const {
        return Color(r + c.r, g + c.g, b + c.b);
    }

    __device__ void operator+=(const Color &c) {
        r += c.r;
        g += c.g;
        b += c.b;
    }

    __device__ Color operator-(const Color &c) const {
        return Color(r - c.r, g - c.g, b - c.b);
    }

    __device__ Color operator*(const Color &c) const {
        return Color(r * c.r, g * c.g, b * c.b);
    }

    __device__ Color operator*(float scalar) const {
        return Color(r * scalar, g * scalar, b * scalar);
    }

    __device__ Color operator/(float divisor) const {
        return Color(r / divisor, g / divisor, b / divisor);
    }

    __host__ __device__ Color clamp() const {
        return Color(single_clamp(r), single_clamp(g), single_clamp(b));
    }

  private:
    __host__ __device__ inline float single_clamp(float x) const {
        return x < 0 ? 0 : (x > 1 ? 1 : x);
    }
};

#endif // CUDA_RAY_TRACER_COLOR_CUH
