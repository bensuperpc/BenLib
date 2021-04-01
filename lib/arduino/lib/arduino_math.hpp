/**
 * @file arduino_math.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-02
 * 
 * MIT License
 * 
 */

#ifndef ARDUINO_MATH_HPP
#define ARDUINO_MATH_HPP
/**
 * @brief 
 *
 * @ingroup Arduino_math
 *
 * @param x 
 * @param a 
 * @param b 
 * @return int 
 */
int constrain(const int x, const int a, const int b);

/**
 * @brief 
 *
 * @ingroup Arduino_math
 * 
 * @param x 
 * @param in_min 
 * @param in_max 
 * @param out_min 
 * @param out_max 
 * @return int 
 */
int map(const int x, const int in_min, const int in_max, const int out_min, const int out_max);

/**
 * @brief 
 *
 * @ingroup Arduino_math
 * 
 * @param x 
 * @param in_min 
 * @param in_max 
 * @param out_min 
 * @param out_max 
 * @return long 
 */
long map(long x, long in_min, long in_max, long out_min, long out_max);

#endif