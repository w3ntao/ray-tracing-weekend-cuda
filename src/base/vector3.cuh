//
// Created by wentao on 4/6/23.
//

#ifndef CUDA_RAY_TRACER_VECTOR3_CUH
#define CUDA_RAY_TRACER_VECTOR3_CUH

#include <stdexcept>

class Vector3 {
  public:
    float x, y, z;

    Vector3() : x(0), y(0), z(0){};

    Vector3(float x, float y, float z);

    Vector3 operator+(const Vector3 &b) const;

    Vector3 operator-(const Vector3 &b) const;

    Vector3 operator-() const;

    Vector3 normalize() const;

    float LengthSquared() const;

    float length() const;

    bool operator==(const Vector3 &b) const;

    bool operator!=(const Vector3 &b) const;

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
        throw std::runtime_error("Vector3: invalid index `" + std::to_string(index) + "`");
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
        throw std::runtime_error("Vector3: invalid index `" + std::to_string(index) + "`");
    }

    int max_axis() const {
        if (x > std::max(y, z)) {
            return 0;
        } else if (y > z) {
            return 1;
        } else {
            return 2;
        }
    }
};

Vector3 operator*(float scalar, const Vector3 &b);

Vector3 operator*(const Vector3 &a, float scalar);

Vector3 operator/(const Vector3 &a, float scalar);

Vector3 cross(const Vector3 &a, const Vector3 &b);

float Dot(const Vector3 &a, const Vector3 &b);

float vector_cosine(const Vector3 &a, const Vector3 &b);

Vector3 min(const Vector3 &a, const Vector3 &b);

Vector3 max(const Vector3 &a, const Vector3 &b);

/*
Point3 operator+(const Point3 &a, const Vector3 &b);

Point3 operator+(const Vector3 &a, const Point3 &b);

Point3 operator-(const Point3 &a, const Vector3 &b);

Point3 operator*(const Float4 &scale, const Point3 &p);
*/

static int MaxDimension(const Vector3 &v) {
    return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
}

static float MaxComponent(const Vector3 &v) {
    return std::max(v.x, std::max(v.y, v.z));
}

static Vector3 Permute(const Vector3 &v, int x, int y, int z) {
    return {v[x], v[y], v[z]};
}

static Vector3 Abs(const Vector3 &v) {
    return Vector3(std::abs(v.x), std::abs(v.y), std::abs(v.z));
}

#endif // CUDA_RAY_TRACER_VECTOR3_CUH
