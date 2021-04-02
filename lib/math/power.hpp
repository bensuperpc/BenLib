/**
 * @file power.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
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
 * @tparam T 
 * @param nbr 
 * @param pow 
 * @return T 
 */
template <typename T> T power(T nbr, long int pow);

/**
 * @brief 
 * 
 * @tparam T 
 * @param nbr 
 * @return true 
 * @return false 
 */
template <typename T> bool isPowerOfTwo(T nbr);
} // namespace math
} // namespace my
#endif
