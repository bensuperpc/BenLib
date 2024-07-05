/**
 * @file sphere.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief sphere header
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */
#ifndef SPHERE_HPP_
#define SPHERE_HPP_

#include "../common/constant.hpp"

#if __cplusplus == 202002L
#include <numbers>
#endif
namespace benlib {

namespace math {

namespace sphere {
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
T sphereVolume(const T& r) {
  return (4.0 / 3.0) * PI * (r * r * r);
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
T sphereSurface(const T& r) {
  return (4.0 * PI * r);
}

}  // namespace sphere
}  // namespace math
}  // namespace benlib

#endif
