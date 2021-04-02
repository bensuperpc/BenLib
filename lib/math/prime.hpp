/**
 * @file prime.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 *
 * @ingroup Math_prime
 *  @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

/*
** https://github.com/niklasb/tcr/blob/master/zahlentheorie/NumberTheory.cpp
** https://stackoverflow.com/questions/4424374/determining-if-a-number-is-prime
** https://www.geeksforgeeks.org/c-program-to-check-prime-number/
** prime.cpp
*/

#ifndef PRIME_HPP_
#define PRIME_HPP_

#include <cmath>

namespace my
{
namespace math
{
namespace prime
{
/**
 * @brief 
 *
 * @ingroup Math_prime
 *  
 * @param n 
 * @return true 
 * @return false 
 */
bool isPrime_opti_0(const long long int &n);

/**
 * @brief 
 *
 * @ingroup Math_prime
 *  
 * @param n 
 * @return true 
 * @return false 
 */
bool isPrime_opti_1(const long long int &n);

/**
 * @brief 
 *
 * @ingroup Math_prime
 *  
 * @param n 
 * @return true 
 * @return false 
 */
bool isPrime_opti_2(const long long int &n);

/**
 * @brief 
 *
 * @ingroup Math_prime
 *  
 * @param n 
 * @return true 
 * @return false 
 */
bool isPrime_opti_3(const long long int &n);

/**
 * @brief 
 *
 * @ingroup Math_prime
 *  
 * @param n 
 * @return true 
 * @return false 
 */
bool isPrime_opti_4(const long long int &n);

/**
 * @brief 
 *
 * @ingroup Math_prime
 *  
 * @param n 
 * @return true 
 * @return false 
 */
bool isPrime_opti_5(const long long int &n);

/**
 * @brief 
 *
 * @ingroup Math_prime
 *  
 * @param int 
 * @param int 
 * @param int 
 * @return long long int 
 */
long long int PowerMod(long long int, long long int, long long int);

/**
 * @brief 
 *
 * @ingroup Math_prime
 *  
 * @param int 
 * @param int 
 * @param int 
 * @return long long int 
 */
long long int MultiplyMod(long long int, long long int, long long int);

/**
 * @brief 
 *
 * @ingroup Math_prime
 *  
 * @tparam T 
 * @return true 
 * @return false 
 */
template <typename T> bool isPrime_opti_5(const T &);

/**
 * @brief 
 *
 * @ingroup Math_prime
 *  
 * @tparam T 
 * @return T 
 */
template <typename T> T PowerMod(T, T, T);

/**
 * @brief 
 *
 * @ingroup Math_prime
 *  
 * @tparam T 
 * @return T 
 */
template <typename T> T MultiplyMod(T, T, T);

/**
 * @brief 
 *
 * @ingroup Math_prime
 *  
 * @param n 
 * @return true 
 * @return false 
 */
bool isPrime_opti_6(const long long int &n);

/**
 * @brief 
 *
 * @ingroup Math_prime
 *  
 * @param n 
 * @return true 
 * @return false 
 */
bool isPrime_opti_7(const long long int &n);

/**
 * @brief 
 *
 * @ingroup Math_prime
 *  
 * @param n 
 * @return true 
 * @return false 
 */
bool isPrime_opti_8(const long long int &n);
} // namespace prime
} // namespace math
} // namespace my
#endif
