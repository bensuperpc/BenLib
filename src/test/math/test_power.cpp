#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE power
/*
extern "C" {
#include "quadmath.h"
}*/

#include <boost/test/unit_test.hpp>
//#include <boost/multiprecision/cpp_int.hpp>
// boost::multiprecision::uint1024_t i = 0;

#include "../../../lib/math/power.hpp"

BOOST_AUTO_TEST_CASE(test_power_1)
{
    BOOST_CHECK_MESSAGE(my::math::power<uint64_t>(2, 16) == 65536, my::math::power<uint64_t>(2, 16) << " instead: " << 65536);
    BOOST_CHECK_MESSAGE(my::math::power<int>(2, 16) == 65536, my::math::power<int>(2, 16) << " instead: " << 65536);
    BOOST_CHECK_MESSAGE(my::math::power<uint64_t>(2, 32) == 4294967296, my::math::power<uint64_t>(2, 32) << " instead: " << 4294967296);
}

BOOST_AUTO_TEST_CASE(test_power_2)
{
    BOOST_CHECK_MESSAGE(my::math::power<uint64_t>(55, 0) == 1, my::math::power<uint64_t>(55, 0) << " instead: " << 1);
    BOOST_CHECK_MESSAGE(my::math::power<uint64_t>(1200, 0) == 1, my::math::power<uint64_t>(1200, 0) << " instead: " << 1);
    BOOST_CHECK_MESSAGE(my::math::power<int>(8000, 0) == 1, my::math::power<int>(8000, 0) << " instead: " << 1);
}

BOOST_AUTO_TEST_CASE(test_power_3)
{
    BOOST_CHECK_MESSAGE(my::math::power<uint64_t>(0, 8) == 0, my::math::power<uint64_t>(0, 8) << " instead: " << 0);
    BOOST_CHECK_MESSAGE(my::math::power<uint64_t>(0, 120) == 0, my::math::power<uint64_t>(0, 120) << " instead: " << 0);
    BOOST_CHECK_MESSAGE(my::math::power<int>(0, 255) == 0, my::math::power<int>(0, 255) << " instead: " << 0);
}

BOOST_AUTO_TEST_CASE(test_power_4)
{
    BOOST_CHECK_MESSAGE(my::math::power<int>(-5, 3) == -125, my::math::power<int>(-5, 3) << " instead: " << -125);
    BOOST_CHECK_MESSAGE(my::math::power<int>(-5, 2) == 25, my::math::power<int>(-5, 2) << " instead: " << 25);
    BOOST_CHECK_MESSAGE(my::math::power<int>(-100, 3) == -1000000, my::math::power<int>(-100, 3) << " instead: " << -1000000);
}