//
// Created by wentao on 4/6/23.
//

#ifndef CUDA_RAY_TRACER_UTIL_H
#define CUDA_RAY_TRACER_UTIL_H

typedef unsigned int uint;

const double CPU_PI = std::acos(-1.0);

inline float between_0_1(float x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

__host__ __device__ inline float gpu_between_0_1(float x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

#endif // CUDA_RAY_TRACER_UTIL_H
