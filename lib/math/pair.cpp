/*
** BENSUPERPC PROJECT, 2020
** Math
** File description:
** pair_imp.hpp
*/

#include "pair.hpp"

template <typename T> bool my::math::is_odd(T nbr)
{
    if (nbr & 1)
        return true;
    else
        return false;
}
template bool my::math::is_odd<int8_t>(int8_t nbr);
template bool my::math::is_odd<int16_t>(int16_t nbr);
template bool my::math::is_odd<int32_t>(int32_t nbr);
template bool my::math::is_odd<int64_t>(int64_t nbr);
template bool my::math::is_odd<uint8_t>(uint8_t nbr);
template bool my::math::is_odd<uint16_t>(uint16_t nbr);
template bool my::math::is_odd<uint32_t>(uint32_t nbr);
template bool my::math::is_odd<uint64_t>(uint64_t nbr);

template <typename T> bool my::math::is_even(T nbr)
{
    if (nbr & 1)
        return false;
    else
        return true;
}
template bool my::math::is_even<int8_t>(int8_t nbr);
template bool my::math::is_even<int16_t>(int16_t nbr);
template bool my::math::is_even<int32_t>(int32_t nbr);
template bool my::math::is_even<int64_t>(int64_t nbr);
template bool my::math::is_even<uint8_t>(uint8_t nbr);
template bool my::math::is_even<uint16_t>(uint16_t nbr);
template bool my::math::is_even<uint32_t>(uint32_t nbr);
template bool my::math::is_even<uint64_t>(uint64_t nbr);