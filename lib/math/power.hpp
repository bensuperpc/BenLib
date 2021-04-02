/**
 * @file power.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 *
 * @ingroup Math_power
 *  @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#ifndef _POWER_HPP_
#define _POWER_HPP_
namespace my
{
namespace math
{
/**
 * @brief 
 *
 * @ingroup Math_power
 *  
 * @tparam T 
 * @param int 
 * @return T 
 */
template <typename T> T power(T, long int);

/**
 * @brief 
 *
 * @ingroup Math_power
 *  
 * @tparam T 
 * @return true 
 * @return false 
 */
template <typename T> bool isPowerOfTwo(T);
} // namespace math
} // namespace my
#endif
