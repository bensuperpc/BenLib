/**
 * @file power_imp.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
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