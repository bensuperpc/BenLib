/**
 * @file pair.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */

#ifndef BENLIB_MATH_PAIR_HPP_
#define BENLIB_MATH_PAIR_HPP_
#include <cstdint>
namespace benlib {
namespace math {

/**
 * @brief
 *
 * @ingroup Math_pair
 *
 * @tparam T
 * @param nbr
 * @return true
 * @return false
 */
template <typename T>
auto isOdd(const T& nbr) -> bool {
    return static_cast<bool>(nbr & 1);
}

/**
 * @brief
 *
 * @ingroup Math_pair
 *
 * @tparam T
 * @param nbr
 * @return true
 * @return false
 */
template <typename T>
auto isEven(const T& nbr) -> bool {
    return !isOdd(nbr);
}
}  // namespace math
}  // namespace benlib
#endif
