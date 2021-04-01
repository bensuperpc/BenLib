/**
 * @file test_getSchwarzschild.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE getSchwarzschild
#include <boost/test/unit_test.hpp>
#include "math/constant.hpp"
#include "math/getSchwarzschild_imp.hpp"

namespace schwarzschild = my::math::schwarzschild;

BOOST_AUTO_TEST_CASE(test_getSchwarzschild_1)
{
    BOOST_CHECK_MESSAGE(static_cast<long int>(schwarzschild::getSchwarzschild<long double>(SUN_MASS)) == static_cast<long int>(2953),
        static_cast<long int>(schwarzschild::getSchwarzschild<long double>(SUN_MASS)) << " instead: " << static_cast<long int>(2953));

    BOOST_CHECK_MESSAGE(static_cast<long int>(schwarzschild::getSchwarzschild<long double>(JUPITER_MASS)) == static_cast<long int>(2),
        static_cast<long int>(schwarzschild::getSchwarzschild<long double>(JUPITER_MASS)) << " instead: " << static_cast<long int>(2));

    BOOST_CHECK_MESSAGE(static_cast<long int>(schwarzschild::getSchwarzschild<long double>(SAGITTARIUS_A_STAR)) == static_cast<long int>(12267767406),
        static_cast<long int>(schwarzschild::getSchwarzschild<long double>(SAGITTARIUS_A_STAR)) << " instead: " << static_cast<long int>(12267767406));

    BOOST_CHECK_MESSAGE(static_cast<long int>(schwarzschild::getSchwarzschild<long double>(TON_618)) == static_cast<long int>(194913974199883),
        static_cast<long int>(schwarzschild::getSchwarzschild<long double>(TON_618)) << " instead: " << static_cast<long int>(194913974199883));
}
BOOST_AUTO_TEST_CASE(test_getSchwarzschild_2)
{
    BOOST_REQUIRE_MESSAGE(static_cast<long int>(schwarzschild::getSchwarzschild<long double>(0)) == static_cast<long int>(0),
        static_cast<long int>(schwarzschild::getSchwarzschild<long double>(0)) << " instead: " << static_cast<long int>(0));
}
/*
BOOST_AUTO_TEST_CASE(my_test2) {
  // seven ways to detect and report the same error:
  BOOST_CHECK(add(2, 2) == 4); // #1 continues on error

  BOOST_REQUIRE(add(2, 2) == 4); // #2 throws on error

  if (add(2, 2) != 4)
    BOOST_ERROR("Ouch..."); // #3 continues on error

  if (add(2, 2) != 4)
    BOOST_FAIL("Ouch..."); // #4 throws on error

  if (add(2, 2) != 4)
    throw "Ouch..."; // #5 throws on error

  BOOST_CHECK_MESSAGE(add(2, 2) == 4, // #6 continues on error
                      "add(..) result: " << add(2, 2));

  BOOST_CHECK_EQUAL(add(2, 2), 4); // #7 continues on error
}
*/
