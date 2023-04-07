//
// Created by wentao on 4/6/23.
//

#ifndef CUDA_RAY_TRACER_RAY_CUH
#define CUDA_RAY_TRACER_RAY_CUH

#include "base/point3.cuh"
#include "base/vector3.cuh"

class Ray {
  public:
    Point3 o;
    Vector3 d;

    __device__ Ray() {}

    __device__ Ray(const Point3 _o, const Vector3 _d) : o(_o), d(_d) {}

    __device__ Point3 at(float t) const {
        return o + t * d;
    }
};

#endif // CUDA_RAY_TRACER_RAY_CUH
