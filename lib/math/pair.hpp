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

#ifndef _PAIR_HPP_
#define _PAIR_HPP_
#include <cstdint>
namespace my
{
namespace math
{

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
template <typename T> bool is_odd(T nbr);

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
template <typename T> bool is_even(T nbr);
} // namespace math
} // namespace my
#endif
