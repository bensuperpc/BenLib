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
** Source: https://stackoverflow.com/questions/178265/what-is-the-most-hard-to-understand-piece-of-c-code-you-know
** https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
** https://en.wikibooks.org/wiki/Optimizing_C%2B%2B/Code_optimization/Faster_operations
*/

#ifndef SQUARE_ROOT_HPP_
#define SQUARE_ROOT_HPP_

#ifdef __FAST_MATH__
#    warning "-ffast-math is broken, don't use it"
#endif

#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>

#define MagicNBR_32 0x5f3759df
#define MagicNBR_64 0x5fe6eb50c7b537a9

namespace math
{

namespace square_root
{
/**
 * @brief 
 *
 * @ingroup Math_square_root
 *  
 * @param x 
 * @return float 
 */
float invsqrt(float x);

/**
 * @brief 
 *
 * @ingroup Math_square_root
 *  
 * @param x 
 * @return double 
 */
double invsqrt(double x);

/**
 * @brief 
 *
 * @ingroup Math_square_root
 *  
 * @param x 
 * @return float 
 */
float sqrt(float x);

/**
 * @brief 
 *
 * @ingroup Math_square_root
 *  
 * @param x 
 * @return double 
 */
double sqrt(double x);

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
template <typename T, char iterations = 2> inline T invsqrt(T nbr);
} // namespace square_root
} // namespace math
#endif
