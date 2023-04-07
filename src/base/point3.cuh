#ifndef CUDA_RAY_TRACER_POINT3_CUH
#define CUDA_RAY_TRACER_POINT3_CUH

#include <stdexcept>

class Vector3;

class Point3 {
  public:
    float x, y, z;

    Point3() : x(0.0), y(0.0), z(0.0){};

    Point3(float _x, float _y, float _z) : x(_x), y(_y), z(_z){};

    float &operator[](int index) {
        if (index == 0) {
            return x;
        }
        if (index == 1) {
            return y;
        }
        if (index == 2) {
            return z;
        }
        throw std::runtime_error("Point3: invalid index `" + std::to_string(index) + "`");
    }

    float operator[](int index) const {
        if (index == 0) {
            return x;
        }
        if (index == 1) {
            return y;
        }
        if (index == 2) {
            return z;
        }
        throw std::runtime_error("Point3: invalid index `" + std::to_string(index) + "`");
    }
};

#endif // CUDA_RAY_TRACER_VEC3_CUH
