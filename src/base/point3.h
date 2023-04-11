#ifndef CUDA_RAY_TRACER_POINT3_H
#define CUDA_RAY_TRACER_POINT3_H

#include <stdexcept>
#include "vector3.h"

class Point3 {
    public:
        float x, y, z;

        __host__ __device__ Point3() : x(0.0), y(0.0), z(0.0){};

        __host__ __device__ Point3(float _x, float _y, float _z) : x(_x), y(_y), z(_z){};

        __host__ __device__ float &operator[](int index) {
            if (index == 0) {
                return x;
            }
            if (index == 1) {
                return y;
            }
            if (index == 2) {
                return z;
            }
#if defined(__CUDA_ARCH__)
            asm("trap;");
#else
            throw std::runtime_error("Point3: invalid index `" + std::to_string(index) + "`");
#endif
        }

        __host__ __device__ float operator[](int index) const {
            if (index == 0) {
                return x;
            }
            if (index == 1) {
                return y;
            }
            if (index == 2) {
                return z;
            }
#if defined(__CUDA_ARCH__)
            asm("trap;");
#else
            throw std::runtime_error("Point3: invalid index `" + std::to_string(index) + "`");
#endif
        }

        __device__ inline Vector3 to_vector() const {
            return Vector3(x, y, z);
        }
};

__device__ inline Point3 operator+(const Point3 &p, const Vector3 &v) {
    return Point3(p.x + v.x, p.y + v.y, p.z + v.z);
}

__device__ inline Vector3 operator-(const Point3 &left, const Point3 &right) {
    return Vector3(left.x - right.x, left.y - right.y, left.z - right.z);
}

__device__ inline Point3 operator-(const Point3 &p, const Vector3 &v) {
    return Point3(p.x - v.x, p.y - v.y, p.z - v.z);
}

__device__ inline Vector3 operator-(const Vector3 &v, const Point3 &p) {
    return Vector3(v.x - p.x, v.y - p.y, v.z - p.z);
}

#endif // CUDA_RAY_TRACER_VEC3_CUH
