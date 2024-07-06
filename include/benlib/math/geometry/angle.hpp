/**
 * @file angle.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief cube header
 * @version 1.0.0
 * @date 2025-07-05
 *
 * MIT License
 *
 */
#ifndef BENLIB_MATH_GEOMETRY_ANGLE_HPP_
#define BENLIB_MATH_GEOMETRY_ANGLE_HPP_

#include <cmath>
#include "../../common/concept.hpp"
#include "../../common/constant.hpp"

namespace benlib {

namespace math {

namespace geometry {

namespace angle {

/**
 * @brief
 *
 * @ingroup Math_geometry
 *
 * @tparam T
 * @param rad
 * @return T
 */

template <ArithmeticType T = double>
static constexpr T radToDeg(T rad) noexcept {
    return rad * 180.0 / PI;
}

/**
 * @brief
 *
 * @ingroup Math_geometry
 *
 * @tparam T
 * @param deg
 * @return T
 */
template <ArithmeticType T = double>
static constexpr T degToRad(T deg) noexcept {
    return deg * PI / 180.0;
}

}  // namespace angle
}  // namespace geometry
}  // namespace math
}  // namespace benlib

#endif
