/*
** BENSUPERPC PROJECT, 2020
** Math
** File description:
** getSchwarzschild.hpp
*/

#include "getSchwarzschild.hpp"

template <typename T> T my::math::schwarzschild::getSchwarzschild(const T &masse)
{
    return (masse > 0) ? (2.0 * CONSTANTE_G * masse) / (pow(LIGHT_SPEED, 2)) : 0;
}
