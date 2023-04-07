//
// Created by wentao on 4/7/23.
//

#ifndef CUDA_RAY_TRACER_SPHERE_H
#define CUDA_RAY_TRACER_SPHERE_H

#include "base/shape.h"

class Sphere : public Shape {
    private:
        Point3 center;
        float radius;

    public:
        __device__ Sphere(const Point3 &_center, float _radius)
            : center(_center), radius(_radius) {}

        __device__ virtual bool intersect(Intersection &intersection, const Ray &ray,
                                          float t_min, float t_max) const {
            Vector3 oc = ray.o - center;
            float a = dot(ray.d, ray.d);
            float b = dot(oc, ray.d);
            float c = dot(oc, oc) - radius * radius;
            float discriminant = b * b - a * c;
            if (discriminant > 0) {
                float temp = (-b - sqrt(discriminant)) / a;
                if (temp < t_max && temp > t_min) {
                    intersection.t = temp;
                    intersection.p = ray.at(intersection.t);
                    intersection.n = (intersection.p - center) / radius;
                    return true;
                }
                temp = (-b + sqrt(discriminant)) / a;
                if (temp < t_max && temp > t_min) {
                    Intersection rec;
                    rec.t = temp;
                    rec.p = ray.at(rec.t);
                    rec.n = (rec.p - center) / radius;
                    return true;
                }
            }
            return false;
        }
};

#endif // CUDA_RAY_TRACER_SPHERE_H
