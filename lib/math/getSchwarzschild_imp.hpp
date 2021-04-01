/**
 * @file getSchwarzschild_imp.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */
#include "getSchwarzschild.hpp"

template <typename T> T my::math::schwarzschild::getSchwarzschild(const T &masse)
{
    return (masse > 0) ? (2.0 * CONSTANTE_G * masse) / (pow(LIGHT_SPEED, 2)) : 0;
}
