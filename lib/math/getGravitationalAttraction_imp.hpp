/**
 * @file getGravitationalAttraction_imp.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#include "getGravitationalAttraction.hpp"

template <typename T> T my::math::ga::getGravitationalAttraction(const T &m1, const T &m2, const T &d)
{
    return (CONSTANTE_G * m1 * m2) / d;
}
