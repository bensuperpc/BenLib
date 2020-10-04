/*
** BENSUPERPC PROJECT, 2020
** Math
** File description:
** prime.hpp
*/

#include "prime.hpp"

bool my::math::prime::isPrime_opti_0(const long long int &n)
{
    if (n <= 1)
        return false;
    for (long long int i = 2; i < n; i++)
        if (n % i == 0)
            return false;

    return true;
}

bool my::math::prime::isPrime_opti_1(const long long int &a)
{
    long long int b = std::sqrt(a);

    for (long long int i = 2; i <= b; i++) {
        if (a % i == 0)
            return false;
    }
    return true;
}

bool my::math::prime::isPrime_opti_2(const long long int &n)
{
    // Corner cases
    if (n <= 1)
        return false;
    if (n <= 3)
        return true;

    if (n % 2 == 0 || n % 3 == 0)
        return false;

    for (long long int i = 5; i * i <= n; i = i + 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return false;

    return true;
}

bool my::math::prime::isPrime_opti_3(const long long int &number)
{
    if (((!(number & 1)) && number != 2) || (number < 2) || (number % 3 == 0 && number != 3))
        return (false);

    for (long long int k = 1; 36 * k * k - 12 * k < number; ++k)
        if ((number % (6 * k + 1) == 0) || (number % (6 * k - 1) == 0))
            return (false);
    return true;
}

bool my::math::prime::isPrime_opti_4(const long long int &number)
{
    if (((!(number & 1)) && number != 2) || (number < 2) || (number % 3 == 0 && number != 3))
        return (false);
    for (long long int k = 1; 36 * k * k - 12 * k < number; ++k)
        if ((number % (6 * k + 1) == 0) || (number % (6 * k - 1) == 0))
            return (false);
    return true;
}

inline long long int my::math::prime::PowerMod(long long int a, long long int n, long long int mod)
{ // computes a^n % mod
    long long int r = 1;
    while (n) {
        if (n & 1)
            r = MultiplyMod(r, a, mod);
        n >>= 1, a = MultiplyMod(a, a, mod);
    }
    return r;
}

inline long long int my::math::prime::MultiplyMod(long long int a, long long int b, long long int mod)
{ // computes a * b % mod
    long long int r = 0;
    a %= mod, b %= mod;
    while (b) {
        if (b & 1)
            r = (r + a) % mod;
        b >>= 1, a = ((long long int)a << 1) % mod;
    }
    return r;
}

// Thank https://github.com/niklasb/tcr/blob/master/zahlentheorie/NumberTheory.cpp
bool my::math::prime::isPrime_opti_5(const long long int &n)
{ // determines if n is a prime number
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
        long long int &&pt = PowerMod(p[i], t, n);
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

bool my::math::prime::isPrime_opti_6(const long long int &number)
{
    if (((!(number & 1)) && number != 2) || (number < 2) || (number % 3 == 0 && number != 3))
        return (false);

    for (long long int k = 1; 36 * k * k - 12 * k < number; ++k)
        if ((number % (6 * k + 1) == 0) || (number % (6 * k - 1) == 0))
            return (false);
    return true;
}

bool my::math::prime::isPrime_opti_7(const long long int &n)
{
    long long int divider = 2;
    while (n % divider != 0) {
        divider++;
    }
    if (n == divider) {
        return true;
    } else {
        return false;
    }
}

bool my::math::prime::isPrime_opti_8(const long long int &n)
{
    return (n < 4000000007) ? isPrime_opti_3(n) : isPrime_opti_5(n);
}
