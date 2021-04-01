/**
 * @file cylinder_imp.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
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
