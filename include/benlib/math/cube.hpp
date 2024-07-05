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
 * @param r
 * @return T
 */
template <typename T>
static constexpr T cubeVolume(const T& w, const T& h, const T& l) {
    return w * h * l;
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
static constexpr T cubeSurface(const T& w, const T& h, const T& l) {
    return 2 * (w * h + h * l + l * w);
}

}  // namespace cube
}  // namespace math
}  // namespace benlib

#endif
