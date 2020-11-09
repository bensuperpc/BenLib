#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE rev_str

#include <boost/predef.h>
#include <boost/test/unit_test.hpp>
extern "C"
{
#include "string_c/my_revstr.h"
}

BOOST_AUTO_TEST_CASE(test_rev_str_1)
{
    char str1[7] = {'B', 'o', 'n', 'j', 'o', 'u', 'r'};
    my_revstr(str1);
    BOOST_REQUIRE_MESSAGE(std::string(str1) == "ruojnoB", std::string(str1) << " Not equal to: ruojnoB");
}

BOOST_AUTO_TEST_CASE(test_rev_str_2)
{
    char str1[7] = {'1', '2', '3', '4', '5', '6', '7'};
    my_revstr(str1);
    BOOST_REQUIRE_MESSAGE(std::string(str1) == "7654321", std::string(str1) << " Not equal to: 7654321");
}

BOOST_AUTO_TEST_CASE(test_rev_str_3)
{
    char str1[4] = {'1', '2', '3', '4'};
    my_revstr(str1);
    BOOST_REQUIRE_MESSAGE(std::string(str1) == "4321", std::string(str1) << " Not equal to: 4321");
}