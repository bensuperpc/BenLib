/**
 * @file cube.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief cube header
 * @version 1.0.0
 * @date 2025-07-05
 *
 * MIT License
 *
 */
#ifndef BENLIB_MATH_CUBE_HPP_
#define BENLIB_MATH_CUBE_HPP_

namespace benlib {

namespace math {

namespace geometry {
/**
 * @brief
 *
 * @ingroup Math_geometry
 *
 * @tparam T
 * @param r
 * @return T
 */
template <typename T>
static constexpr T distance2D(const T& x1, const T& y1, const T& x2, const T& y2) {
    return std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
}

/**
 * @brief
 *
 * @ingroup Math_cube
 *
 * @tparam T
 * @param r
 * @return T
 */
template <typename T>
static constexpr T distance3D(const T& x1, const T& y1, const T& z1, const T& x2, const T& y2, const T& z2) {
    return std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2) + std::pow(z2 - z1, 2));
}

}  // namespace geometry
}  // namespace math
}  // namespace benlib

#endif
