/**
 * @file prime.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
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

#ifndef BENLIB_MATH_PRIME_HPP_
#define BENLIB_MATH_PRIME_HPP_

#include <cmath>

namespace benlib {
namespace math {
namespace prime {
/**
 * @brief
 *
 * @ingroup Math_prime
 *
 * @param n
 * @return true
 * @return false
 */
bool isPrime_opti_0(const long long int& number) {
    if (((!(number & 1)) && number != 2) || (number < 2) || (number % 3 == 0 && number != 3))
        return (false);

    for (long long int k = 1; 36 * k * k - 12 * k < number; ++k)
        if ((number % (6 * k + 1) == 0) || (number % (6 * k - 1) == 0))
            return (false);
    return true;
}

/**
 * @brief
 *
 * @ingroup Math_prime
 *
 * @param a
 * @param b
 * @param c
 * @return long long int
 */
inline long long int MultiplyMod(long long int a, long long int b,
                                 long long int mod) {  // computes a * b % mod
    long long int r = 0;
    a %= mod, b %= mod;
    while (b) {
        if (b & 1)
            r = (r + a) % mod;
        b >>= 1, a = ((long long int)a << 1) % mod;
    }
    return r;
}

/**
 * @brief
 *
 * @ingroup Math_prime
 *
 * @param a
 * @param b
 * @param c
 * @return long long int
 */
inline long long int PowerMod(long long int a, long long int n,
                              long long int mod) {  // computes a^n % mod
    long long int r = 1;
    while (n) {
        if (n & 1)
            r = MultiplyMod(r, a, mod);
        n >>= 1, a = MultiplyMod(a, a, mod);
    }
    return r;
}

/**
 * @brief
 *
 * @ingroup Math_prime
 *
 * @tparam T
 * @param nbr
 * @return true
 * @return false
 */
// Thanks to
// https://github.com/niklasb/tcr/blob/master/zahlentheorie/NumberTheory.cpp
bool isPrime_opti_1(const long long int& n) {  // determines if n is a prime number
    const long long int pn = 9, p[] = {2, 3, 5, 7, 11, 13, 17, 19, 23};
    for (long long int i = 0; i < pn; ++i)
        if (n % p[i] == 0)
            return n == p[i];
    if (n < p[pn - 1])
        return 0;
    long long int s = 0, t = n - 1;
    while (~t & 1)
        t >>= 1, ++s;
    for (long long int i = 0; i < pn; ++i) {
        long long int&& pt = PowerMod(p[i], t, n);
        if (pt == 1)
            continue;
        bool ok = 0;
        for (long long int j = 0; j < s && !ok; ++j) {
            if (pt == n - 1)
                ok = 1;
            pt = MultiplyMod(pt, pt, n);
        }
        if (!ok)
            return 0;
    }
    return 1;
}

/**
 * @brief
 *
 * @ingroup Math_prime
 *
 * @param n
 * @return true
 * @return false
 */
bool isPrime_opti_8(const long long int& n) {
    return (n < 4000000007) ? isPrime_opti_0(n) : isPrime_opti_1(n);
}
}  // namespace prime
}  // namespace math
}  // namespace benlib
#endif
