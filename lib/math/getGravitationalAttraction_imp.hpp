/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** getGravitationalAttraction.cpp
*/

#include "getGravitationalAttraction.hpp"

template <typename T> T my::math::ga::getGravitationalAttraction(const T &m1, const T &m2, const T &d)
{
    return (CONSTANTE_G * m1 * m2) / d;
}
