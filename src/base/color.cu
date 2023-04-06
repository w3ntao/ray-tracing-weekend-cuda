//
// Created by wentao on 4/6/23.
//

#include "color.cuh"

RGBColor RGBColor::operator+(const RGBColor &c) const {
    return RGBColor(r + c.r, g + c.g, b + c.b);
}

RGBColor RGBColor::operator-(const RGBColor &c) const {
    return RGBColor(r - c.r, g - c.g, b - c.b);
}

RGBColor RGBColor::operator*(const RGBColor &c) const {
    return RGBColor(r * c.r, g * c.g, b * c.b);
}

bool RGBColor::operator==(const RGBColor &c) const {
    return r == c.r && g == c.g && b == c.b;
}

bool RGBColor::operator!=(const RGBColor &b) const {
    return !(*this == b);
}

RGBColor RGBColor::clamp() const {
    return RGBColor(single_clamp(r), single_clamp(g), single_clamp(b));
}

RGBColor operator*(float scalar, const RGBColor &c) {
    return RGBColor(scalar * c.r, scalar * c.g, scalar * c.b);
}

RGBColor operator*(const RGBColor &c, float scalar) {
    return scalar * c;
}

void RGBColor::operator+=(const RGBColor &c) {
    r += c.r;
    g += c.g;
    b += c.b;
}

RGBColor operator/(const RGBColor &c, float scalar) {
    return RGBColor(c.r / scalar, c.g / scalar, c.b / scalar);
}