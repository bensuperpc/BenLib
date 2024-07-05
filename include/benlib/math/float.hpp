/**
 * @file float.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */

#ifndef FLOAT_HPP_
#define FLOAT_HPP_

#include <cmath>
#include <limits>

#include "../common/concept.hpp"

namespace benlib {
namespace math {
namespace fp {
/**
 * @brief Compare two float if they are equal
 *
 * @tparam T
 * @tparam relativeFloat
 * @param f1
 * @param f2
 * @return true
 * @return false
 */

template <typename T, bool relative = true>
static constexpr auto areEqual(T f1, T f2) ->
    typename std::enable_if<std::is_floating_point<T>::value, bool>::type {
  if constexpr (relative) {
    return (std::fabs(f1 - f2) <= std::numeric_limits<T>::epsilon() *
                                      std::fmax(std::fabs(f1), std::fabs(f2)));
  } else {
    return (std::fabs(f1 - f2) <= std::numeric_limits<T>::epsilon());
  }
}

}  // namespace fp
}  // namespace math
}  // namespace benlib
#endif
