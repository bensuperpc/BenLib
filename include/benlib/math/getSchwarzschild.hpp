/**
 * @file getSchwarzschild.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */
#ifndef BENLIB_MATH_GETSCHWARZCHILD_HPP_
#define BENLIB_MATH_GETSCHWARZCHILD_HPP_

#include <cmath>

#include "../common/concept.hpp"
#include "../common/constant.hpp"

namespace benlib {
namespace math {
namespace schwarzschild {
/**
 * @brief Get the Schwarzschild object
 *
 * @ingroup Math_Schwarzschild
 *
 * @tparam T
 * @param masse
 * @return T
 */
template <ArithmeticType T>
auto getSchwarzschild(const T& masse) noexcept -> T {
    return (masse > 0) ? (2.0 * CONSTANTE_G * masse) / (std::pow(LIGHT_SPEED, 2)) : 0;
}
}  // namespace schwarzschild
}  // namespace math
}  // namespace benlib

#endif
