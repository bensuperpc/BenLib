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
#define BOOST_TEST_MODULE string_CRC

#include <boost/predef.h>
#include <boost/test/unit_test.hpp>
#include "crypto/crypto.hpp"

BOOST_AUTO_TEST_CASE(test_crypto_CRC_1)
{
    const std::string str = "Bonjour";
    const auto &&crc = my::crypto::GetCrc32(str);
    BOOST_REQUIRE(crc == 0x6A0BC954);
    const auto &&jam = my::crypto::GetJAMCRC32(str);
    BOOST_REQUIRE(jam == 0x95F436AB);
}

BOOST_AUTO_TEST_CASE(test_crypto_CRC_2)
{
    const std::string str = "Il est 16h30, c'est la fin des cours !";
    const auto &&crc = my::crypto::GetCrc32(str);
    BOOST_REQUIRE(crc == 0x15BF4A09);
    const auto &&jam = my::crypto::GetJAMCRC32(str);
    BOOST_REQUIRE(jam == 0xEA40B5F6);
}

BOOST_AUTO_TEST_CASE(test_crypto_CRC_3)
{
    const std::string str = "I'm king of the world ! :D";
    const auto &&crc = my::crypto::GetCrc32(str);
    BOOST_REQUIRE(crc == 0xC3212D08);
    const auto &&jam = my::crypto::GetJAMCRC32(str);
    BOOST_REQUIRE(jam == 0x3CDED2F7);
}

BOOST_AUTO_TEST_CASE(test_crypto_CRC_4)
{
    const std::string str = "+-*/$~#[](){}&=/:;?;.!";
    const auto &&crc = my::crypto::GetCrc32(str);
    BOOST_REQUIRE(crc == 0xEAFC7C67);
    // const auto &&jam = my::crypto::GetJAMCRC32(str);
    // BOOST_REQUIRE((~jam) == 0x345FBABC);
}

BOOST_AUTO_TEST_CASE(test_crypto_CRC_5)
{
    const std::string str = "Where is Brian? Brian is in the kitchen";
    const auto &&crc = my::crypto::GetCrc32(str);
    BOOST_REQUIRE(crc == 0x60ECAB66);
    // const auto &&jam = my::crypto::GetJAMCRC32(str);
    // BOOST_REQUIRE((~jam) == -0x60ecab67);
}