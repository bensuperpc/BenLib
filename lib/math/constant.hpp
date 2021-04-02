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

/** @defgroup Math Mathematic
 *  @brief The main Math group who contain all software to do math :)
 */
/** @defgroup Math_prime Math prime
 *  @ingroup Math
 *  @brief All you need for calc and test prime numbers
 *  @sa @link Math The first group Math@endlink
 */
/** @defgroup Math_power Math power
 *  @ingroup Math
 *  @brief Math power calculation
 *  @sa @link Math The first group Math@endlink
 */
/** @defgroup Math_count_digits Math count digits
 *  @ingroup Math
 *  @brief count digits in variable
 *  @sa @link Math The first group Math@endlink
 */




#ifndef CONSTANT_HPP_
#define CONSTANT_HPP_
//#define BUILD_BUG_ON(condition) ((void)sizeof(char[1 - 2*!!(condition)]))
// BUILD_BUG_ON((sizeof(struct mystruct) % 8) != 0);

#ifdef CMAKE_CXX_EXTENSIONS
#    if CMAKE_CXX_EXTENSIONS == 1
#        define O_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280O
#        define Q_PI 3.1415926535897932384626433832795028841971693993751058Q
#    endif
#endif

/// Define Pi (Double)
#define PI 3.141592653589793238462643383279502884L

/// Define Pi (Float)
#define F_PI 3.14159265358979323846

/// Define light speed constant (in m/s)
#define LIGHT_SPEED 299792458

/// Define Gravitational constant
#define CONSTANTE_G 6.67408e-11

/// Define Earth mass (in Kg)
#define EARTH_MASS 5.972e24

/// Define Sun mass (in Kg)
#define SUN_MASS 1.98847e30

/// Define Jupiter mass (in Kg)
#define JUPITER_MASS 1.89813e27

/// Define Earth-Sun distance (in m)
#define EARTH_SUN_DISTANCE 149597870e3

/// Define Earth radius (in m)
#define EARTH_RADIUS 6.3781 * 10e6

/// Define Moon radius (in m)
#define MOON_RADIUS 1737.4e3

/// Define SAGITTARIUS mass (in m)
#define SAGITTARIUS_A_STAR 4.154e6 * SUN_MASS
/// Define ?
#define TON_618 6.6e10 * SUN_MASS

#endif