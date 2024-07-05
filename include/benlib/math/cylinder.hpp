/**
 * @file cylinder.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */

#ifndef BENLIB_MATH_CYLINDER_HPP_
#define BENLIB_MATH_CYLINDER_HPP_

#include "../common/constant.hpp"

namespace benlib {
namespace math {

namespace cylinder {

/**
 * @brief
 *
 * @ingroup Math_cylinder
 *
 * @tparam T
 * @param r
 * @param h
 * @return T
 */
template <typename T>
static constexpr T cylinderSurface(const T& r, const T& h) noexcept {
    return 2.0 * PI * r * r + 2.0 * PI * r * h;
}

/**
 * @brief
 *
 * @ingroup Math_cylinder
 *
 * @tparam T
 * @param r
 * @param h
 * @return T
 */
template <typename T>
static constexpr auto cylinderVolume(const T& r, const T& h) noexcept -> T {
    return h * PI * r * r;
}

}  // namespace cylinder
}  // namespace math
}  // namespace benlib

#endif
