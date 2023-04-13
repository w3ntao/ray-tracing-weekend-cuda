//
// Created by wentao on 4/13/23.
//

#ifndef CUDA_RAY_TRACER_PATH_H
#define CUDA_RAY_TRACER_PATH_H

#include "base/integrator.h"
#include <cfloat>

class PathIntegrator : public Integrator {
    public:
        PathIntegrator() = default;

        __device__ Color get_radiance(const Ray &ray, const World *const *world,
                                      curandState *local_rand_state) const override {
            Ray current_ray = ray;
            Color current_attenuation = Color(1.0, 1.0, 1.0);
            for (int i = 0; i < 50; i++) {
                Intersection intersection;
                if ((*world)->intersect(intersection, current_ray, 0.001f, FLT_MAX)) {
                    Ray scattered_ray;
                    Color attenuation;
                    if (!intersection.material_ptr->scatter(current_ray, intersection, attenuation,
                                                            scattered_ray, local_rand_state)) {
                        return Color(0.0, 0.0, 0.0);
                    }

                    current_attenuation *= attenuation;
                    current_ray = scattered_ray;
                    continue;
                }

                float t = 0.5f * (current_ray.d.normalize().y + 1.0f);
                Vector3 c = (1.0f - t) * Vector3(1.0, 1.0, 1.0) + t * Vector3(0.5, 0.7, 1.0);
                return current_attenuation * Color(c.x, c.y, c.z);
            }

            return Color(0.0, 0.0, 0.0); // exceeded recursion
        }
};

#endif // CUDA_RAY_TRACER_PATH_H
