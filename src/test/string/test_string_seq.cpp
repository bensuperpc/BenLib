//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  BenLib, 2021                                            //
//  Created: 03, March, 2021                                //
//  Modified: 03, March, 2021                               //
//  file: crypto.cpp                                        //
//  Crypto                                                  //
//  Source:                                                 //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_string_seq

#include <boost/predef.h>
#include <boost/test/unit_test.hpp>
#include "string_lib/string_lib_impl.hpp"

BOOST_AUTO_TEST_CASE(test_string_seq_0)
{
    const char alpha[27] = alphabetMax; // Get alphabetic seq
    char tmp1[2] = {0};
    char tmp2[2] = {'A', '\0'};
    for (std::size_t i = 1; i < 26; i++) {
        my::string::findStringInv<std::size_t>(i, tmp1);
        tmp2[0] = alpha[i - 1];
        BOOST_REQUIRE(strcmp(tmp1, tmp2) == 0);
    }
}

BOOST_AUTO_TEST_CASE(test_string_seq_1)
{
    char tmp1[2] = {0};
    char tmp2[2] = {'A', '\0'};

    my::string::findStringInv<std::size_t>(1, tmp1);
    BOOST_REQUIRE(strcmp(tmp1, tmp2) == 0);
}

BOOST_AUTO_TEST_CASE(test_string_seq_2)
{
    char tmp1[2] = {0};
    char tmp2[2] = {'Z', '\0'};

    my::string::findStringInv<std::size_t>(26, tmp1);
    BOOST_REQUIRE(strcmp(tmp1, tmp2) == 0);
}

BOOST_AUTO_TEST_CASE(test_string_seq_3)
{
    char tmp1[3] = {0};
    const char tmp2[3] = {'A', 'A', '\0'};

    my::string::findStringInv<std::size_t>(27, tmp1);
    BOOST_REQUIRE(strcmp(tmp1, tmp2) == 0);
}

BOOST_AUTO_TEST_CASE(test_string_seq_4)
{
    char tmp1[3] = {0};
    const char tmp2[3] = {'Y', 'Y', '\0'};

    my::string::findStringInv<std::size_t>(675, tmp1);
    BOOST_REQUIRE(strcmp(tmp1, tmp2) == 0);
}

BOOST_AUTO_TEST_CASE(test_string_seq_5)
{
    char tmp1[3] = {0};
    const char tmp2[3] = {'Z', 'Z', '\0'};

    my::string::findStringInv<std::size_t>(702, tmp1); // 26^1 + 26^2
    BOOST_REQUIRE(strcmp(tmp1, tmp2) == 0);
}

BOOST_AUTO_TEST_CASE(test_string_seq_6)
{
    char tmp1[4] = {0};
    const char tmp2[4] = {'Z', 'Z', 'Z', '\0'};

    my::string::findStringInv<std::size_t>(18278, tmp1); // 26^1 + 26^2 + 26^3
    BOOST_REQUIRE(strcmp(tmp1, tmp2) == 0);
}

BOOST_AUTO_TEST_CASE(test_string_seq_7)
{
    char tmp1[5] = {0};
    const char tmp2[5] = {'Z', 'Z', 'Z', 'Z', '\0'};

    my::string::findStringInv<std::size_t>(475254, tmp1); // 26^1 + 26^2 + 26^3 + 26^4
    BOOST_REQUIRE(strcmp(tmp1, tmp2) == 0);
}

BOOST_AUTO_TEST_CASE(test_string_seq_8)
{
    char tmp1[6] = {0};
    const char tmp2[6] = {'A', 'A', 'A', 'A', 'A','\0'};

    my::string::findStringInv<std::size_t>(475255, tmp1); // 26^1 + 26^2 + 26^3 + 26^4
    BOOST_REQUIRE(strcmp(tmp1, tmp2) == 0);
}

