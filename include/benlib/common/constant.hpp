/**
 * @file constant.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */

#ifndef CONSTANT_HPP_
#define CONSTANT_HPP_

#if __cplusplus >= 202002L
#include <numbers>
#endif

#ifdef CMAKE_CXX_EXTENSIONS
#if CMAKE_CXX_EXTENSIONS == 1
#define O_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280O
#define Q_PI 3.1415926535897932384626433832795028841971693993751058Q
#endif
#endif

#if __cplusplus < 202002L
// Pi (Double)
#define PI 3.141592653589793238462643383279502884L
#else
#define PI std::numbers::pi
#endif

// Pi (Float)
#define F_PI 3.14159265358979323846F

// light speed constant (in m/s)
#define LIGHT_SPEED 299792458

// Gravitational constant
#define CONSTANTE_G 6.67408e-11

// Earth mass (in Kg)
#define EARTH_MASS 5.972e24

// Sun mass (in Kg)
#define SUN_MASS 1.98847e30

// Jupiter mass (in Kg)
#define JUPITER_MASS 1.89813e27

// Earth-Sun distance (in m)
#define EARTH_SUN_DISTANCE 149597870e3

// Earth radius (in m)
#define EARTH_RADIUS 6.3781 * 10e6

// Moon radius (in m)
#define MOON_RADIUS 1737.4e3

// SAGITTARIUS mass (in m)
#define SAGITTARIUS_A_STAR 4.154e6 * SUN_MASS

#endif