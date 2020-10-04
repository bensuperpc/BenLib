/*
** BENSUPERPC PROJECT, 2020
** Math
** F
** prime_imp.hpp.cpp
*/

#include "prime.hpp"

template <typename T> inline T my::math::prime::PowerMod(T a, T n, T mod)
{ // computes a^n % mod
    T r = 1;
    while (n) {
        if (n & 1)
            r = MultiplyMod<T>(r, a, mod);
        n >>= 1, a = MultiplyMod<T>(a, a, mod);
    }
    return r;
}

template <typename T> inline T my::math::prime::MultiplyMod(T a, T b, T mod)
{ // computes a * b % mod
    T r = 0;
    a %= mod, b %= mod;
    while (b) {
        if (b & 1)
            r = (r + a) % mod;
        b >>= 1, a = ((T)a << 1) % mod;
    }
    return r;
}

// Thank https://github.com/niklasb/tcr/blob/master/zahlentheorie/NumberTheory.cpp
template <typename T> bool my::math::prime::isPrime_opti_5(const T &n)
{ // determines if n is a prime number
    const T pn = 9, p[] = {2, 3, 5, 7, 11, 13, 17, 19, 23};
    for (T i = 0; i < pn; ++i)
        if (n % p[i] == 0)
            return n == p[i];
    if (n < p[pn - 1])
        return 0;
    T s = 0, t = n - 1;
    while (~t & 1)
        t >>= 1, ++s;
    for (T i = 0; i < pn; ++i) {
        T &&pt = PowerMod<T>(p[i], t, n);
        if (pt == 1)
            continue;
        bool ok = 0;
        for (T j = 0; j < s && !ok; ++j) {
            if (pt == n - 1)
                ok = 1;
            pt = MultiplyMod<T>(pt, pt, n);
        }
        if (!ok)
            return 0;
    }
    return 1;
}
