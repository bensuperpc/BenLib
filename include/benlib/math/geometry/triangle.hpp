/**
 * @file triangle.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief triangle header
 * @version 1.0.0
 * @date 2025-07-05
 *
 * MIT License
 *
 */
#ifndef BENLIB_MATH_GEOMETRY_TRIANGLE_HPP_
#define BENLIB_MATH_GEOMETRY_TRIANGLE_HPP_

#include "../../common/constant.hpp"

namespace benlib {

namespace math {

namespace geometry {

namespace triangle {

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
static constexpr T triangleSurface(const T& a, const T& b, const T& c) noexcept {
    T s = (a + b + c) / 2;
    return std::sqrt(s * (s - a) * (s - b) * (s - c));
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
static constexpr T triangleRectangleSurface(const T& b, const T& h) noexcept {
    return (b * h) / 2;
}

}  // namespace triangle
}  // namespace geometry
}  // namespace math
}  // namespace benlib

#endif
