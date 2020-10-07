/*
** BENSUPERPC PROJECT, 2020
** Math
** File description:
** cylinder.hpp
*/

#include "cylinder.hpp"

#if __cplusplus < 202002L
template <typename T> T my::math::cylinder::cylinderVolume(const T &r, const T &h)
{
    return r * r * h * PI;
}

template <typename T> T my::math::cylinder::cylinderSurface(const T &r, const T &h)
{
    return 2.0 * PI * r * r + 2.0 * PI * r * h;
}
#else
template <typename T> T my::math::cylinder::cylinderVolume(const T &r, const T &h)
{
    return r * r * h * std::numbers::pi;
}

template <typename T> T my::math::cylinder::cylinderSurface(const T &r, const T &h)
{
    return 2.0 * std::numbers::pi * r * r + 2.0 * std::numbers::pi * r * h;
}
#endif
