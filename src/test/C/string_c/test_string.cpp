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