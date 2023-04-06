#include "vector3.cuh"
#include "point3.cuh"

Vector3::Vector3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

Vector3 Vector3::operator+(const Vector3 &b) const {
    return Vector3(x + b.x, y + b.y, z + b.z);
}

Vector3 Vector3::operator-(const Vector3 &b) const {
    return Vector3(x - b.x, y - b.y, z - b.z);
}

Vector3 Vector3::operator-() const {
    return Vector3(-x, -y, -z);
}

Vector3 Vector3::normalize() const {
    return *this / length();
}

Vector3 operator*(float scalar, const Vector3 &b) {
    return Vector3(b.x * scalar, b.y * scalar, b.z * scalar);
}

Vector3 operator*(const Vector3 &a, float scalar) {
    return scalar * a;
}

Vector3 operator/(const Vector3 &a, float scalar) {
    return Vector3(a.x / scalar, a.y / scalar, a.z / scalar);
}

Vector3 cross(const Vector3 &a, const Vector3 &b) {
    return Vector3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float Dot(const Vector3 &a, const Vector3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float vector_cosine(const Vector3 &a, const Vector3 &b) {
    return Dot(a.normalize(), b.normalize());
}

float Vector3::LengthSquared() const {
    return Dot(*this, *this);
}

float Vector3::length() const {
    return std::sqrt(LengthSquared());
}

bool Vector3::operator==(const Vector3 &b) const {
    return x == b.x && y == b.y && z == b.z;
}

bool Vector3::operator!=(const Vector3 &b) const {
    return !(*this == b);
}

Vector3 min(const Vector3 &a, const Vector3 &b) {
    return Vector3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

Vector3 max(const Vector3 &a, const Vector3 &b) {
    return Vector3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

Point3 operator+(const Point3 &a, const Vector3 &b) {
    return Point3(a.x + b.x, a.y + b.y, a.z + b.z);
}

Point3 operator+(const Vector3 &a, const Point3 &b) {
    return b + a;
}

Point3 operator-(const Point3 &a, const Vector3 &b) {
    return Point3(a.x - b.x, a.y - b.y, a.z - b.z);
}
