/**
 * @file getGravitationalAttraction.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */

#ifndef BENLIB_MATH_GETGRAVITATIONATTRACTION_HPP
#define BENLIB_MATH_GETGRAVITATIONATTRACTION_HPP

#include "../common/constant.hpp"

namespace benlib {
namespace math {
/**
 * @brief Math_GravitationalAttraction
 * @namespace ga
 */
namespace ga {
/**
 * @brief Get the Gravitational Attraction object
 *
 * @ingroup Math_GravitationalAttraction
 *
 * @tparam T
 * @param m1
 * @param m2
 * @param d
 * @return T
 */
template <typename T>
auto getGravitationalAttraction(const T& m1, const T& m2, const T& d) -> T {
    return (CONSTANTE_G * m1 * m2) / d;
}
}  // namespace ga
}  // namespace math
}  // namespace benlib

#endif
