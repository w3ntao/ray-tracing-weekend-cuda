//
// Created by wentao on 4/12/23.
//

#ifndef CUDA_RAY_TRACER_MATERIAL_H
#define CUDA_RAY_TRACER_MATERIAL_H

struct Intersection;

#include "base/ray.h"
#include "base/shape.h"

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const Vector3 &v, const Vector3 &n, float ni_over_nt, Vector3 &refracted) {
    Vector3 uv = v.normalize();
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    } else
        return false;
}

__device__ Vector3 random_vector(curandState *local_rand_state) {
    return Vector3(curand_uniform(local_rand_state), curand_uniform(local_rand_state),
                   curand_uniform(local_rand_state));
}

__device__ Vector3 random_in_unit_sphere(curandState *local_rand_state) {
    Vector3 p;
    do {
        p = 2.0f * random_vector(local_rand_state) - Vector3(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ Vector3 reflect(const Vector3 &v, const Vector3 &n) {
    return v - 2.0f * dot(v, n) * n;
}

class Material {
    public:
        __device__ virtual bool scatter(const Ray &r_in, const Intersection &rec,
                                        Color &attenuation, Ray &scattered,
                                        curandState *local_rand_state) const = 0;
};

class lambertian : public Material {
    public:
        __device__ lambertian(const Color &a) : albedo(a) {}
        __device__ virtual bool scatter(const Ray &r_in, const Intersection &rec,
                                        Color &attenuation, Ray &scattered,
                                        curandState *local_rand_state) const {
            Point target = rec.p + rec.n + random_in_unit_sphere(local_rand_state);
            scattered = Ray(rec.p, target - rec.p);
            attenuation = albedo;
            return true;
        }

        Color albedo;
};

class metal : public Material {
    public:
        __device__ metal(const Color &a, float f) : albedo(a) {
            if (f < 1)
                fuzz = f;
            else
                fuzz = 1;
        }
        __device__ virtual bool scatter(const Ray &r_in, const Intersection &rec,
                                        Color &attenuation, Ray &scattered,
                                        curandState *local_rand_state) const {
            Vector3 reflected = reflect(r_in.d.normalize(), rec.n);
            scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            return (dot(scattered.d, rec.n) > 0.0f);
        }
        Color albedo;
        float fuzz;
};

class dielectric : public Material {
    public:
        __device__ dielectric(float ri) : ref_idx(ri) {}
        __device__ virtual bool scatter(const Ray &r_in, const Intersection &rec,
                                        Color &attenuation, Ray &scattered,
                                        curandState *local_rand_state) const {
            Vector3 outward_normal;
            Vector3 reflected = reflect(r_in.d, rec.n);
            float ni_over_nt;
            attenuation = Color(1.0, 1.0, 1.0);
            Vector3 refracted;
            float reflect_prob;
            float cosine;
            if (dot(r_in.d, rec.n) > 0.0f) {
                outward_normal = -rec.n;
                ni_over_nt = ref_idx;
                cosine = dot(r_in.d, rec.n) / r_in.d.length();
                cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
            } else {
                outward_normal = rec.n;
                ni_over_nt = 1.0f / ref_idx;
                cosine = -dot(r_in.d, rec.n) / r_in.d.length();
            }
            if (refract(r_in.d, outward_normal, ni_over_nt, refracted))
                reflect_prob = schlick(cosine, ref_idx);
            else
                reflect_prob = 1.0f;
            if (curand_uniform(local_rand_state) < reflect_prob)
                scattered = Ray(rec.p, reflected);
            else
                scattered = Ray(rec.p, refracted);
            return true;
        }

        float ref_idx;
};

#endif // CUDA_RAY_TRACER_MATERIAL_H
