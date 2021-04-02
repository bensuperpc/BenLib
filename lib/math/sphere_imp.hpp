/**
 * @file sphere_imp.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief sphere template implementation
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */
#include "sphere.hpp"
template <typename T> T my::math::sphere::sphereVolume(const T &r)
{
    return (4.0 / 3.0) * PI * (r * r * r);
}
template <typename T> T my::math::sphere::sphereSurface(const T &r)
{
    return (4.0 * PI * r);
}

#if __cplusplus < 202002L
#    ifdef CMAKE_CXX_EXTENSIONS
#        if CMAKE_CXX_EXTENSIONS == 1
template <typename T> T my::math::sphere::sphereVolume_Q(const T &r)
{
#            pragma GCC diagnostic ignored "-Wpedantic"
    return (4.0 / 3.0) * Q_PI * (r * r * r);
}
template <typename T> T my::math::sphere::sphereSurface_Q(const T &r)
{
#            pragma GCC diagnostic ignored "-Wpedantic"
    return (4.0 * Q_PI * r);
}
#        endif
#    endif
#else
template <typename T> T my::math::sphere::sphereVolume_Q(const T &r)
{
    return (4.0 / 3.0) * std::numbers::pi * (r * r * r);
}
template <typename T> T my::math::sphere::sphereSurface_Q(const T &r)
{
    return (4.0 * std::numbers::pi * r);
}
#endif
