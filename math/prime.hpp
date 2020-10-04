/*
** BENSUPERPC PROJECT, 2020
** Math
** F
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
bool isPrime_opti_0(const long long int &);
bool isPrime_opti_1(const long long int &);
bool isPrime_opti_2(const long long int &);
bool isPrime_opti_3(const long long int &);
bool isPrime_opti_4(const long long int &);

bool isPrime_opti_5(const long long int &);
inline long long int PowerMod(long long int, long long int, long long int);
inline long long int MultiplyMod(long long int, long long int, long long int);

template <typename T> inline bool isPrime_opti_5(const T &);
template <typename T> inline T PowerMod(T, T, T);
template <typename T> inline T MultiplyMod(T, T, T);

bool isPrime_opti_6(const long long int &);
bool isPrime_opti_7(const long long int &);
bool isPrime_opti_8(const long long int &);

#include "prime_imp.hpp"
} // namespace prime
} // namespace math
} // namespace my
// THANK https://www.geeksforgeeks.org/c-program-to-check-prime-number/
// THANK https://stackoverflow.com/questions/4424374/determining-if-a-number-is-prime
#endif
