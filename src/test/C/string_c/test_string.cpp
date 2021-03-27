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
    char str1[8] = {'B', 'o', 'n', 'j', 'o', 'u', 'r', '\0'};
    my_revstr(str1);
    BOOST_REQUIRE_MESSAGE(std::string(str1) == "ruojnoB", std::string(str1) << " Not equal to: ruojnoB");
}

BOOST_AUTO_TEST_CASE(test_rev_str_2)
{
    char str1[8] = {'1', '2', '3', '4', '5', '6', '7', '\0'};
    my_revstr(str1);
    BOOST_REQUIRE_MESSAGE(std::string(str1) == "7654321", std::string(str1) << " Not equal to: 7654321");
}

BOOST_AUTO_TEST_CASE(test_rev_str_3)
{
    char str1[3] = {'1', '2', '\0'};
    my_revstr(str1);
    BOOST_REQUIRE_MESSAGE(std::string(str1) == "21", std::string(str1) << " Not equal to: 21");
}

BOOST_AUTO_TEST_CASE(test_rev_str_4)
{
    char str1[4] = {'1', '2', '3', '\0'};
    my_revstr(str1);
    BOOST_REQUIRE_MESSAGE(std::string(str1) == "321", std::string(str1) << " Not equal to: 321");
}

BOOST_AUTO_TEST_CASE(test_rev_str_5)
{
    char str1[4] = {'.', '.', '.', '\0'};
    my_revstr(str1);
    BOOST_REQUIRE_MESSAGE(std::string(str1) == "...", std::string(str1) << " Not equal to: ...");
}