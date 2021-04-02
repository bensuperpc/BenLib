/**
 * @file pair.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */
#include "pair.hpp"

template <typename T> bool my::math::is_odd(T nbr)
{
    if (nbr & 1)
        return true;
    else
        return false;
}

template <typename T> bool my::math::is_even(T nbr)
{
    if (nbr & 1)
        return false;
    else
        return true;
}