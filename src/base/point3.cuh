#ifndef CUDA_RAY_TRACER_POINT3_CUH
#define CUDA_RAY_TRACER_POINT3_CUH

#include <stdexcept>

class Vector3;

class Point3 {
  public:
    float x, y, z;

    Point3() = default;

    Point3(float x, float y, float z);
    
    Vector3 operator-(const Point3 &b) const;

    bool operator==(const Point3 &b) const;

    bool operator!=(const Point3 &b) const;

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

Point3 operator+(const Point3 &a, const Point3 &b);

Point3 operator*(float scalar, const Point3 &b);

Point3 operator*(const Point3 &a, float scalar);

Point3 operator/(const Point3 &a, float divisor);

Point3 min(const Point3 &a, const Point3 &b);

Point3 max(const Point3 &a, const Point3 &b);

static Point3 Permute(const Point3 &p, int x, int y, int z) {
    return {p[x], p[y], p[z]};
}

#endif // CUDA_RAY_TRACER_VEC3_CUH
