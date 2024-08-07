/**
 * @file sphere.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief sphere header
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */
#ifndef BENLIB_MATH_GEOMETRY_SPHERE_HPP_
#define BENLIB_MATH_GEOMETRY_SPHERE_HPP_

#include "../../common/constant.hpp"

namespace benlib {

namespace math {

namespace geometry {

namespace sphere {
/**
 * @brief
 *
 * @ingroup Math_sphere
 *
 * @tparam T
 * @param r
 * @return T
 */
template <typename T>
static constexpr T sphereVolume(const T& r) noexcept {
    return (4.0 / 3.0) * PI * (r * r * r);
}

/**
 * @brief
 *
 * @ingroup Math_sphere
 *
 * @tparam T
 * @param r
 * @return T
 */
template <typename T>
static constexpr T sphereSurface(const T& r) noexcept {
    return (4.0 * PI * r * r);
}

}  // namespace sphere
}  // namespace geometry
}  // namespace math
}  // namespace benlib

#endif
