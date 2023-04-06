//
// Created by wentao on 4/6/23.
//

#ifndef CUDA_RAY_TRACER_COLOR_CUH
#define CUDA_RAY_TRACER_COLOR_CUH

class RGBColor {
  public:
    float r, g, b;

    RGBColor() {}

    RGBColor(float r, float g, float b) : r(r), g(g), b(b) {}

    static RGBColor rep(float v) {
        return RGBColor(v, v, v);
    }

    RGBColor operator+(const RGBColor &c) const;

    void operator+=(const RGBColor &c);

    RGBColor operator-(const RGBColor &c) const;

    RGBColor operator*(const RGBColor &c) const;

    bool operator==(const RGBColor &b) const;

    bool operator!=(const RGBColor &b) const;

    RGBColor clamp() const;

    RGBColor gamma(float gam) const;

    float luminance() const;

  private:
    inline float single_clamp(float x) const {
        return x < 0 ? 0 : (x > 1 ? 1 : x);
    }
};

RGBColor operator*(float scalar, const RGBColor &b);

RGBColor operator*(const RGBColor &a, float scalar);

RGBColor operator/(const RGBColor &a, float scalar);

#endif // CUDA_RAY_TRACER_COLOR_CUH
