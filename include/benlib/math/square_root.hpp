/**
 * @file square_root.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief square root template header
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */

/*
** Source:
*https://stackoverflow.com/questions/178265/what-is-the-most-hard-to-understand-piece-of-c-code-you-know
** https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
** https://en.wikibooks.org/wiki/Optimizing_C%2B%2B/Code_optimization/Faster_operations
*/

#ifndef SQUARE_ROOT_HPP_
#define SQUARE_ROOT_HPP_

#ifdef __FAST_MATH__
#warning "-ffast-math is broken, don't use it"
#endif

#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>

#define MagicNBR_32 0x5f3759df
#define MagicNBR_64 0x5fe6eb50c7b537a9

namespace math {

namespace square_root {

/**
 * @brief
 *
 * @ingroup Math_square_root
 *
 * @tparam T
 * @tparam iterations
 * @param nbr
 * @return T
 */
template <typename T, char iterations = 2>
auto invsqrt(const T& nbr) -> T;

// THANK https://stackoverflow.com/a/59248244/10152334
template <typename T, char iterations>
T math::square_root::invsqrt(const T& x) {
  static_assert(std::is_floating_point<T>::value, "T must be floating point");
  static_assert(iterations == 1 or iterations == 2,
                "itarations must equal 1 or 2");
  typedef typename std::conditional<sizeof(T) == 8, std::int64_t,
                                    std::int32_t>::type Tint;
  T y = x;
  T x2 = y * 0.5;
  Tint i = *(Tint*)&y;
  i = (sizeof(T) == 8 ? MagicNBR_64 : MagicNBR_32) - (i >> 1);
  y = *(T*)&i;
  y = y * (1.5 - (x2 * y * y));
  if (iterations == 2)
    y = y * (1.5 - (x2 * y * y));
  return y;
}

/**
 * @brief
 *
 * @ingroup Math_square_root
 *
 * @param x
 * @return float
 */
auto invsqrt(const float& nbr) -> float {
  float x = nbr;
  float xhalf = 0.5f * x;
  int i = *(int*)&x;
  i = MagicNBR_32 - (i >> 1);
  x = *(float*)&i;
  x = x * (1.5f - xhalf * x * x);
  // x  = x * (1.5 - ( xhalf * x * x ));   // 2nd iteration, this can be removed
  return x;
}

/**
 * @brief
 *
 * @ingroup Math_square_root
 *
 * @param x
 * @return double
 */
auto invsqrt(const double& x) -> double {
  double y = x;
  double x2 = y * 0.5;
  std::int64_t i = *(std::int64_t*)&y;
  i = MagicNBR_64 - (i >> 1);
  y = *(double*)&i;
  y = y * (1.5 - (x2 * y * y));
  // y  = y * ( 1.5 - ( x2 * y * y ) );   // 2nd iteration, this can be removed
  return y;
}

/**
 * @brief
 *
 * @ingroup Math_square_root
 *
 * @param x
 * @return float
 */
auto sqrt(const float& x) -> float {
  return std::sqrt(x);
}

/**
 * @brief
 *
 * @ingroup Math_square_root
 *
 * @param x
 * @return double
 */
auto sqrt(const double& x) -> double {
  return std::sqrt(x);
}

}  // namespace square_root
}  // namespace math
#endif
