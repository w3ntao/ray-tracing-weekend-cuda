//
// Created by wentao on 4/7/23.
//

#ifndef CUDA_RAY_TRACER_SHAPE_H
#define CUDA_RAY_TRACER_SHAPE_H

#include <optional>
#include "base/ray.cuh"

class Intersection {
    public:
        float t;
        Point3 p;
        Vector3 n;
};

class Shape {
    public:
        __device__ virtual bool intersect(Intersection &intersection, const Ray &ray,
                                          float t_min, float t_max) const = 0;
};

#endif // CUDA_RAY_TRACER_SHAPE_H
