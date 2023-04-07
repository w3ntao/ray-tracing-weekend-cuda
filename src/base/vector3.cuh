//
// Created by wentao on 4/6/23.
//

#ifndef CUDA_RAY_TRACER_VECTOR3_CUH
#define CUDA_RAY_TRACER_VECTOR3_CUH

#include <stdexcept>

class Vector3 {
  public:
    float x, y, z;

    __host__ __device__ Vector3() : x(0), y(0), z(0){};

    __host__ __device__ Vector3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

    __host__ __device__ Vector3 operator+(const Vector3 &b) const {
        return Vector3(x + b.x, y + b.y, z + b.z);
    }

    __host__ __device__ Vector3 operator-(const Vector3 &b) const {
        return Vector3(x - b.x, y - b.y, z - b.z);
    }

    __host__ __device__ Vector3 operator-() const {
        return Vector3(-x, -y, -z);
    }

    __host__ __device__ Vector3 operator*(float factor) const {
        return Vector3(x * factor, y * factor, z * factor);
    }

    __host__ __device__ Vector3 operator/(float divisor) const {
        return Vector3(x / divisor, y / divisor, z / divisor);
    }

    __host__ __device__ float dot(const Vector3 &v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    __host__ __device__ float length() const {
        return std::sqrt(this->dot(*this));
    }

    __host__ __device__ Vector3 normalize() const {
        return *this / length();
    }

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
        throw std::runtime_error("Vector3: invalid index `" + std::to_string(index) + "`");
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
        throw std::runtime_error("Vector3: invalid index `" + std::to_string(index) + "`");
#endif
    }
};
__host__ __device__ Vector3 operator*(float factor, const Vector3 &v) {
    return v * factor;
}

#endif // CUDA_RAY_TRACER_VECTOR3_CUH
