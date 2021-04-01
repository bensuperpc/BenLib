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

#include <algorithm>
#include <cmath>
extern "C"
{
#include <stdio.h>
}

namespace my
{
namespace math
{
namespace fp
{

/**
 * @brief 
 * 
 * @param x 
 * @param y 
 * @return true 
 * @return false 
 */
bool are_aqual(double &x, double &y);

} // namespace fp
} // namespace math
} // namespace my
#endif
