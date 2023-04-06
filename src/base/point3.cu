#include <point3.cuh>
#include <vector3.cuh>

Point3::Point3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

Vector3 Point3::operator-(const Point3 &b) const {
    return Vector3(x - b.x, y - b.y, z - b.z);
}

bool Point3::operator==(const Point3 &b) const {
    return x == b.x && y == b.y && z == b.z;
}

bool Point3::operator!=(const Point3 &b) const {
    return !(*this == b);
}

Point3 operator+(const Point3 &a, const Point3 &b) {
    return Point3(a.x + b.x, a.y + b.y, a.z + b.z);
}

Point3 operator*(float scalar, const Point3 &b) {
    return Point3(b.x * scalar, b.y * scalar, b.z * scalar);
}

Point3 operator*(const Point3 &a, float scalar) {
    return scalar * a;
}

Point3 operator/(const Point3 &a, float divisor) {
    return a * (1.0 / divisor);
}

Point3 min(const Point3 &a, const Point3 &b) {
    return Point3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

Point3 max(const Point3 &a, const Point3 &b) {
    return Point3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}
