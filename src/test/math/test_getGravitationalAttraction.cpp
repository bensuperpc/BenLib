/**
 * @file test_getGravitationalAttraction.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE getGravitationalAttraction

#include <boost/test/unit_test.hpp>
#include "math/constant.hpp"
#include "math/getGravitationalAttraction_imp.hpp"

BOOST_AUTO_TEST_CASE(test_getGravitationalAttraction_1)
{
    BOOST_CHECK_MESSAGE(static_cast<long int>(my::math::ga::getGravitationalAttraction<long double>(10000, 20000, 30000)) == static_cast<long int>(0),
        static_cast<long int>(my::math::ga::getGravitationalAttraction<long double>(10000, 20000, 30000)) << " instead: " << static_cast<long int>(0));
}
BOOST_AUTO_TEST_CASE(test_getGravitationalAttraction_2)
{
    BOOST_CHECK_MESSAGE(static_cast<long int>(my::math::ga::getGravitationalAttraction<long double>(0, 0, 1)) == static_cast<long int>(0),
        static_cast<long int>(my::math::ga::getGravitationalAttraction<long double>(0, 0, 1)) << " instead: " << static_cast<long int>(0));
}