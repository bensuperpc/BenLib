/**
 * @file sphere.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */
#ifndef SPHERE_HPP_
#define SPHERE_HPP_

#include "constant.hpp"

#if __cplusplus == 202002L
#    include <numbers>
#endif
namespace my
{

namespace math
{

namespace sphere
{
/**
 * @brief 
 * 
 * @tparam T 
 * @param r 
 * @return T 
 */
template <typename T> T sphereSurface(const T &r);

/**
 * @brief 
 * 
 * @tparam T 
 * @param r 
 * @return T 
 */
template <typename T> T sphereVolume(const T &r);

/**
 * @brief 
 * 
 * @tparam T 
 * @param r 
 * @return T 
 */
template <typename T> T sphereSurface_Q(const T &r);

/**
 * @brief 
 * 
 * @tparam T 
 * @param r 
 * @return T 
 */
template <typename T> T sphereVolume_Q(const T &r);

} // namespace sphere
} // namespace math
} // namespace my

#endif
