#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE my_str_isnum

#include <boost/predef.h>
#include <boost/test/unit_test.hpp>
extern "C"
{
#include "string_is/my_str_isnum.h"
}

BOOST_AUTO_TEST_CASE(test_my_str_isnum_1)
{
    const char str[] = "";
    BOOST_REQUIRE_MESSAGE(my_str_isnum(str) == 0, my_str_isnum(str) << " Not equal to: 0");
}

BOOST_AUTO_TEST_CASE(test_my_str_isnum_2)
{
    const char str[] = "0123456789";
    BOOST_REQUIRE_MESSAGE(my_str_isnum(str) == 1, my_str_isnum(str) << " Not equal to: 1");
}

BOOST_AUTO_TEST_CASE(test_my_str_isnum_3)
{
    const char str[] = "9876543210";
    BOOST_REQUIRE_MESSAGE(my_str_isnum(str) == 1, my_str_isnum(str) << " Not equal to: 1");
}

BOOST_AUTO_TEST_CASE(test_my_str_isnum_4)
{
    const char str[] = ".";
    BOOST_REQUIRE_MESSAGE(my_str_isnum(str) == 0, my_str_isnum(str) << " Not equal to: 0");
}

BOOST_AUTO_TEST_CASE(test_my_str_isnum_5)
{
    const char str[] = "515.445";
    BOOST_REQUIRE_MESSAGE(my_str_isnum(str) == 0, my_str_isnum(str) << " Not equal to: 0");
}

BOOST_AUTO_TEST_CASE(test_my_str_isnum_6)
{
    const char str[] = "515a445";
    BOOST_REQUIRE_MESSAGE(my_str_isnum(str) == 0, my_str_isnum(str) << " Not equal to: 0");
}

BOOST_AUTO_TEST_CASE(test_my_str_isnum_7)
{
    const char str[] = "515Z445";
    BOOST_REQUIRE_MESSAGE(my_str_isnum(str) == 0, my_str_isnum(str) << " Not equal to: 0");
}

BOOST_AUTO_TEST_CASE(test_my_str_isnum_8)
{
    const char str[] = "515|445";
    BOOST_REQUIRE_MESSAGE(my_str_isnum(str) == 0, my_str_isnum(str) << " Not equal to: 0");
}

BOOST_AUTO_TEST_CASE(test_my_str_isnum_9)
{
    const char str[] = "5";
    BOOST_REQUIRE_MESSAGE(my_str_isnum(str) == 1, my_str_isnum(str) << " Not equal to: 1");
}