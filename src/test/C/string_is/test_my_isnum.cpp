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
    BOOST_REQUIRE_MESSAGE(my_isnum('8') == 1, my_isnum('8') << " Not equal to: 1");
}