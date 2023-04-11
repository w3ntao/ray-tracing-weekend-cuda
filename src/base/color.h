//
// Created by wentao on 4/6/23.
//

#ifndef CUDA_RAY_TRACER_COLOR_H
#define CUDA_RAY_TRACER_COLOR_H

class Color {
    public:
        float r, g, b;

        __host__ __device__ Color() : r(0.0), g(0.0), b(0.0) {}

        __host__ __device__ Color(float r, float g, float b) : r(r), g(g), b(b) {}

        __host__ __device__ Color clamp() const {
            return Color(single_clamp(r), single_clamp(g), single_clamp(b));
        }

        __device__ inline void operator+=(const Color &c) {
            r += c.r;
            g += c.g;
            b += c.b;
        }

        __device__ inline void operator*=(const Color &c) {
            r *= c.r;
            g *= c.g;
            b *= c.b;
        }

        __device__ inline void operator/=(float divisor) {
            r /= divisor;
            g /= divisor;
            b /= divisor;
        }

    private:
        __host__ __device__ inline float single_clamp(float x) const {
            return x < 0 ? 0 : (x > 1 ? 1 : x);
        }
};

__device__ inline Color operator+(const Color &left, const Color &right) {
    return Color(left.r + right.r, left.g + right.g, left.b + right.b);
}

__device__ inline Color operator-(const Color &left, const Color &right) {
    return Color(left.r - right.r, left.g - right.g, left.b - right.b);
}

__device__ inline Color operator*(const Color &left, const Color &right) {
    return Color(left.r * right.r, left.g * right.g, left.b * right.b);
}

__device__ inline Color operator*(const Color &c, float scalar) {
    return Color(c.r * scalar, c.g * scalar, c.b * scalar);
}

__device__ inline Color operator*(float scalar, const Color &c) {
    return Color(c.r * scalar, c.g * scalar, c.b * scalar);
}

__device__ inline Color operator/(const Color c, float divisor) {
    return Color(c.r / divisor, c.g / divisor, c.b / divisor);
}

#endif // CUDA_RAY_TRACER_COLOR_H
