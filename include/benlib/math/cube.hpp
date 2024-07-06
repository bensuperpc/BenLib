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

#include "../common/constant.hpp"

namespace benlib {

namespace math {

namespace cube {
/**
 * @brief
 *
 * @ingroup Math_cube
 *
 * @tparam T
 * @param width
 * @param height
 * @param length
 * @return T
 */
template <typename T>
static constexpr T cubeVolume(const T& width, const T& height, const T& length) noexcept {
    return width * height * length;
}

/**
 * @brief
 *
 * @ingroup Math_sphere
 *
 * @tparam T
 * @param width
 * @param height
 * @param length
 * @return T
 */
template <typename T>
static constexpr T cubeSurface(const T& width, const T& height, const T& length) noexcept {
    return 2 * (width * height + height * length + length * width);
}

}  // namespace cube
}  // namespace math
}  // namespace benlib

#endif
