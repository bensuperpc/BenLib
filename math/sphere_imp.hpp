/*
** BENSUPERPC PROJECT, 2020
** Math
** File description:
** sphere.hpp
*/

#include "sphere.hpp"
template <typename T> T my::math::sphere::sphereVolume(const T &r)
{
    return (4.0 / 3.0) * PI * (r * r * r);
}
template <typename T> T my::math::sphere::sphereSurface(const T &r)
{
    return 4.0 * PI * r;
}

#if __cplusplus < 202002L
#    if CMAKE_CXX_EXTENSIONS == 1
template <typename T> T my::math::sphere::sphereVolume_Q(const T &r)
{
#        pragma GCC diagnostic ignored "-Wpedantic"
    return (4.0 / 3.0) * Q_PI * (r * r * r);
}
template <typename T> T my::math::sphere::sphereSurface_Q(const T &r)
{
#        pragma GCC diagnostic ignored "-Wpedantic"
    return 4.0 * Q_PI * r;
}
#    endif
#else
template <typename T> T my::math::sphere::sphereVolume_Q(const T &r)
{
    return (4.0 / 3.0) * std::numbers::pi * (r * r * r);
}
template <typename T> T my::math::sphere::sphereSurface_Q(const T &r)
{
    return 4.0 * std::numbers::pi * r;
}
#endif
