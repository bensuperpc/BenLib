/*
** BENSUPERPC PROJECT, 2020
** Math
** File description:
** power_imp.hpp
*/

#include "power.hpp"

template <typename T> T my::math::power(T nb, long int p)
{
    if (p < 0)
        return (0);
    if (p != 0)
        return (nb * power(nb, p - 1));
    else
        return 1;
}

template <typename T> bool my::math::isPowerOfTwo(T x)
{
    return x && (!(x & (x - 1)));
}