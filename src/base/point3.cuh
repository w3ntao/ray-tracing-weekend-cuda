#ifndef CUDA_RAY_TRACER_POINT3_CUH
#define CUDA_RAY_TRACER_POINT3_CUH

#include <stdexcept>
#include "vector3.cuh"

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

        __host__ __device__ Point3 operator+(const Vector3 &v) const {
            return Point3(x + v.x, y + v.y, z + v.z);
        }

        __host__ __device__ Vector3 operator-(const Point3 &p) const {
            return Vector3(x - p.x, y - p.y, z - p.z);
        }

        __host__ __device__ Vector3 to_vector() const {
            return Vector3(x, y, z);
        }
};

#endif // CUDA_RAY_TRACER_VEC3_CUH
