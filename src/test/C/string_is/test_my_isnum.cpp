#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE my_isnum

#include <boost/predef.h>
#include <boost/test/unit_test.hpp>
extern "C"
{
#include "string_is/my_isnum.h"
}

BOOST_AUTO_TEST_CASE(test_my_isnum_1)
{
    const char str = '0';
    BOOST_REQUIRE_MESSAGE(my_isnum(str) == 1, my_isnum(str) << " Not equal to: 1");
}

BOOST_AUTO_TEST_CASE(test_my_isnum_2)
{
    const char str = '5';
    BOOST_REQUIRE_MESSAGE(my_isnum(str) == 1, my_isnum(str) << " Not equal to: 1");
}

BOOST_AUTO_TEST_CASE(test_my_isnum_3)
{
    const char str = '9';
    BOOST_REQUIRE_MESSAGE(my_isnum(str) == 1, my_isnum(str) << " Not equal to: 1");
}

BOOST_AUTO_TEST_CASE(test_my_isnum_letter_1)
{
    const char str = 'a';
    BOOST_REQUIRE_MESSAGE(my_isnum(str) == 0, my_isnum(str) << " Not equal to: 0");
}

BOOST_AUTO_TEST_CASE(test_my_isnum_letter_2)
{
    const char str = 'A';
    BOOST_REQUIRE_MESSAGE(my_isnum(str) == 0, my_isnum(str) << " Not equal to: 0");
}

BOOST_AUTO_TEST_CASE(test_my_isnum_letter_3)
{
    const char str = 'z';
    BOOST_REQUIRE_MESSAGE(my_isnum(str) == 0, my_isnum(str) << " Not equal to: 0");
}

BOOST_AUTO_TEST_CASE(test_my_isnum_letter_4)
{
    const char str = 'Z';
    BOOST_REQUIRE_MESSAGE(my_isnum(str) == 0, my_isnum(str) << " Not equal to: 0");
}

BOOST_AUTO_TEST_CASE(test_my_isnum_char_1)
{
    const char str = '?';
    BOOST_REQUIRE_MESSAGE(my_isnum(str) == 0, my_isnum(str) << " Not equal to: 0");
}

BOOST_AUTO_TEST_CASE(test_my_isnum_char_2)
{
    const char str = '|';
    BOOST_REQUIRE_MESSAGE(my_isnum(str) == 0, my_isnum(str) << " Not equal to: 0");
}

BOOST_AUTO_TEST_CASE(test_my_isnum_char_3)
{
    const char str = '.';
    BOOST_REQUIRE_MESSAGE(my_isnum(str) == 0, my_isnum(str) << " Not equal to: 0");
}

BOOST_AUTO_TEST_CASE(test_my_isnum_char_4)
{
    const char str = '_';
    BOOST_REQUIRE_MESSAGE(my_isnum(str) == 0, my_isnum(str) << " Not equal to: 0");
}