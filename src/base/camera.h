//
// Created by wentao on 4/11/23.
//

#ifndef CUDA_RAY_TRACER_CAMERA_H
#define CUDA_RAY_TRACER_CAMERA_H

#include "base/ray.h"

class Camera {
    public:
        __device__ Camera() {
            lower_left_corner = Vector3(-2.0, -1.0, -1.0);
            horizontal = Vector3(4.0, 0.0, 0.0);
            vertical = Vector3(0.0, 2.0, 0.0);
            origin = Point3(0.0, 0.0, 0.0);
        }
        __device__ Ray get_ray(float u, float v) {
            return Ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
        }

        Point3 origin;
        Vector3 lower_left_corner;
        Vector3 horizontal;
        Vector3 vertical;
};

#endif // CUDA_RAY_TRACER_CAMERA_H
