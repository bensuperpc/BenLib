/**
 * @file random.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

/*
** Source: https://stackoverflow.com/questions/14638739/generating-a-random-double-between-a-range-of-values
*/

#ifndef RANDOM_HPP_
#define RANDOM_HPP_

#include <algorithm>
#include <ctime>
#include <iostream>
#include <iterator>
#include <random>

namespace my
{
namespace math
{
namespace rand
{
/**
 * @brief 
 * 
 * @param fMin 
 * @param fMax 
 * @return double 
 */
double rand(double fMin, double fMax);
} // namespace rand
} // namespace math
} // namespace my
#endif
