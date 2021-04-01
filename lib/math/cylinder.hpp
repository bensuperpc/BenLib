/**
 * @file cylinder.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#ifndef CYLINDER_HPP_
#define CYLINDER_HPP_

#include "constant.hpp"

#if __cplusplus == 202002L
#    include <numbers>
#endif

namespace my
{
namespace math
{

namespace cylinder
{

/**
 * @brief 
 * 
 * @tparam T 
 * @param r 
 * @param h 
 * @return T 
 */
template <typename T> T cylinderSurface(const T &r, const T &h);

/**
 * @brief 
 * 
 * @tparam T 
 * @param r 
 * @param h 
 * @return T 
 */
template <typename T> T cylinderVolume(const T &r, const T &h);
} // namespace cylinder
} // namespace math
} // namespace my

#endif
