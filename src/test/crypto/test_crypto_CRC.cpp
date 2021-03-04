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
#include "crypto/crypto_CRC32.hpp"

BOOST_AUTO_TEST_CASE(test_crypto_CRC_1)
{
    const std::string str = "Bonjour";
    const uint32_t CRC_VALUE = 0x6A0BC954;
    const uint32_t JAMCRC_VALUE = 0x95F436AB;

    const auto &&crc_boost = my::crypto::CRC32_Boost(str);
    BOOST_REQUIRE(crc_boost == CRC_VALUE);
    const auto &&crc_stackoverflow = my::crypto::CRC32_StackOverflow((const unsigned char *)str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_stackoverflow == CRC_VALUE);
    const auto &&crc_1byte_tableless = my::crypto::CRC32_1byte_tableless(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_1byte_tableless == CRC_VALUE);
    const auto &&crc_1byte_tableless2 = my::crypto::CRC32_1byte_tableless(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_1byte_tableless2 == CRC_VALUE);
    const auto &&crc_1byte = my::crypto::CRC32_1byte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_1byte == CRC_VALUE);
    const auto &&crc_bitwise = my::crypto::CRC32_bitwise(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_bitwise == CRC_VALUE);
    const auto &&crc_halfbyte = my::crypto::CRC32_halfbyte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_halfbyte == CRC_VALUE);
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_8
    const auto &&crc_4bytes = my::crypto::CRC32_4bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_4bytes == CRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_8
    const auto &&crc_8bytes = my::crypto::CRC32_8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_8bytes == CRC_VALUE);
    const auto &&crc_4x8bytes = my::crypto::CRC32_4x8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_4x8bytes == CRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_16
    const auto &&crc_16bytes = my::crypto::CRC32_16bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_16bytes == CRC_VALUE);
    const auto &&crc_16bytes_prefetch = my::crypto::CRC32_16bytes_prefetch(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_16bytes_prefetch == CRC_VALUE);
#endif

    const auto &&jam_boost = my::crypto::JAMCRC_Boost(str);
    BOOST_REQUIRE(jam_boost == JAMCRC_VALUE);
    const auto &&jam_stackoverflow = my::crypto::JAMCRC_StackOverflow((const unsigned char *)str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_stackoverflow == JAMCRC_VALUE);
    const auto &&jam_1byte_tableless = my::crypto::JAMCRC_1byte_tableless(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_1byte_tableless == JAMCRC_VALUE);
    const auto &&jam_1byte_tableless2 = my::crypto::JAMCRC_1byte_tableless2(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_1byte_tableless2 == JAMCRC_VALUE);
    const auto &&jam_1byte = my::crypto::JAMCRC_1byte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_1byte == JAMCRC_VALUE);
    const auto &&jam_bitwise = my::crypto::JAMCRC_bitwise(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_bitwise == JAMCRC_VALUE);
    const auto &&jam_halfbyte = my::crypto::JAMCRC_halfbyte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_halfbyte == JAMCRC_VALUE);
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_4
    const auto &&jam_4bytes = my::crypto::JAMCRC_4bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_4bytes == JAMCRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_8
    const auto &&jam_8bytes = my::crypto::JAMCRC_8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_8bytes == JAMCRC_VALUE);
    const auto &&jam_4x8bytes = my::crypto::JAMCRC_4x8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_4x8bytes == JAMCRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_16
    const auto &&jam_16bytes = my::crypto::JAMCRC_16bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_16bytes == JAMCRC_VALUE);
    const auto &&jam_16bytes_prefetch = my::crypto::JAMCRC_16bytes_prefetch(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_16bytes_prefetch == JAMCRC_VALUE);
#endif
}

BOOST_AUTO_TEST_CASE(test_crypto_CRC_2)
{
    const std::string str = "Il est 16h30, c'est la fin des cours !";
    const uint32_t CRC_VALUE = 0x15BF4A09;
    const uint32_t JAMCRC_VALUE = 0xEA40B5F6;

    const auto &&crc_boost = my::crypto::CRC32_Boost(str);
    BOOST_REQUIRE(crc_boost == CRC_VALUE);
    const auto &&crc_stackoverflow = my::crypto::CRC32_StackOverflow((const unsigned char *)str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_stackoverflow == CRC_VALUE);
    const auto &&crc_1byte_tableless = my::crypto::CRC32_1byte_tableless(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_1byte_tableless == CRC_VALUE);
    const auto &&crc_1byte_tableless2 = my::crypto::CRC32_1byte_tableless(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_1byte_tableless2 == CRC_VALUE);
    const auto &&crc_1byte = my::crypto::CRC32_1byte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_1byte == CRC_VALUE);
    const auto &&crc_bitwise = my::crypto::CRC32_bitwise(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_bitwise == CRC_VALUE);
    const auto &&crc_halfbyte = my::crypto::CRC32_halfbyte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_halfbyte == CRC_VALUE);
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_8
    const auto &&crc_4bytes = my::crypto::CRC32_4bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_4bytes == CRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_8
    const auto &&crc_8bytes = my::crypto::CRC32_8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_8bytes == CRC_VALUE);
    const auto &&crc_4x8bytes = my::crypto::CRC32_4x8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_4x8bytes == CRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_16
    const auto &&crc_16bytes = my::crypto::CRC32_16bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_16bytes == CRC_VALUE);
    const auto &&crc_16bytes_prefetch = my::crypto::CRC32_16bytes_prefetch(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_16bytes_prefetch == CRC_VALUE);
#endif

    const auto &&jam_boost = my::crypto::JAMCRC_Boost(str);
    BOOST_REQUIRE(jam_boost == JAMCRC_VALUE);
    const auto &&jam_stackoverflow = my::crypto::JAMCRC_StackOverflow((const unsigned char *)str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_stackoverflow == JAMCRC_VALUE);
    const auto &&jam_1byte_tableless = my::crypto::JAMCRC_1byte_tableless(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_1byte_tableless == JAMCRC_VALUE);
    const auto &&jam_1byte_tableless2 = my::crypto::JAMCRC_1byte_tableless2(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_1byte_tableless2 == JAMCRC_VALUE);
    const auto &&jam_1byte = my::crypto::JAMCRC_1byte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_1byte == JAMCRC_VALUE);
    const auto &&jam_bitwise = my::crypto::JAMCRC_bitwise(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_bitwise == JAMCRC_VALUE);
    const auto &&jam_halfbyte = my::crypto::JAMCRC_halfbyte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_halfbyte == JAMCRC_VALUE);
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_4
    const auto &&jam_4bytes = my::crypto::JAMCRC_4bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_4bytes == JAMCRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_8
    const auto &&jam_8bytes = my::crypto::JAMCRC_8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_8bytes == JAMCRC_VALUE);
    const auto &&jam_4x8bytes = my::crypto::JAMCRC_4x8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_4x8bytes == JAMCRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_16
    const auto &&jam_16bytes = my::crypto::JAMCRC_16bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_16bytes == JAMCRC_VALUE);
    const auto &&jam_16bytes_prefetch = my::crypto::JAMCRC_16bytes_prefetch(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_16bytes_prefetch == JAMCRC_VALUE);
#endif
}

BOOST_AUTO_TEST_CASE(test_crypto_CRC_3)
{
    const std::string str = "I'm king of the world ! :D";
    const uint32_t CRC_VALUE = 0xC3212D08;
    const uint32_t JAMCRC_VALUE = 0x3CDED2F7;

    const auto &&crc_boost = my::crypto::CRC32_Boost(str);
    BOOST_REQUIRE(crc_boost == CRC_VALUE);
    const auto &&crc_stackoverflow = my::crypto::CRC32_StackOverflow((const unsigned char *)str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_stackoverflow == CRC_VALUE);
    const auto &&crc_1byte_tableless = my::crypto::CRC32_1byte_tableless(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_1byte_tableless == CRC_VALUE);
    const auto &&crc_1byte_tableless2 = my::crypto::CRC32_1byte_tableless(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_1byte_tableless2 == CRC_VALUE);
    const auto &&crc_1byte = my::crypto::CRC32_1byte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_1byte == CRC_VALUE);
    const auto &&crc_bitwise = my::crypto::CRC32_bitwise(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_bitwise == CRC_VALUE);
    const auto &&crc_halfbyte = my::crypto::CRC32_halfbyte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_halfbyte == CRC_VALUE);
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_8
    const auto &&crc_4bytes = my::crypto::CRC32_4bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_4bytes == CRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_8
    const auto &&crc_8bytes = my::crypto::CRC32_8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_8bytes == CRC_VALUE);
    const auto &&crc_4x8bytes = my::crypto::CRC32_4x8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_4x8bytes == CRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_16
    const auto &&crc_16bytes = my::crypto::CRC32_16bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_16bytes == CRC_VALUE);
    const auto &&crc_16bytes_prefetch = my::crypto::CRC32_16bytes_prefetch(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_16bytes_prefetch == CRC_VALUE);
#endif

    const auto &&jam_boost = my::crypto::JAMCRC_Boost(str);
    BOOST_REQUIRE(jam_boost == JAMCRC_VALUE);
    const auto &&jam_stackoverflow = my::crypto::JAMCRC_StackOverflow((const unsigned char *)str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_stackoverflow == JAMCRC_VALUE);
    const auto &&jam_1byte_tableless = my::crypto::JAMCRC_1byte_tableless(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_1byte_tableless == JAMCRC_VALUE);
    const auto &&jam_1byte_tableless2 = my::crypto::JAMCRC_1byte_tableless2(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_1byte_tableless2 == JAMCRC_VALUE);
    const auto &&jam_1byte = my::crypto::JAMCRC_1byte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_1byte == JAMCRC_VALUE);
    const auto &&jam_bitwise = my::crypto::JAMCRC_bitwise(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_bitwise == JAMCRC_VALUE);
    const auto &&jam_halfbyte = my::crypto::JAMCRC_halfbyte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_halfbyte == JAMCRC_VALUE);
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_4
    const auto &&jam_4bytes = my::crypto::JAMCRC_4bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_4bytes == JAMCRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_8
    const auto &&jam_8bytes = my::crypto::JAMCRC_8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_8bytes == JAMCRC_VALUE);
    const auto &&jam_4x8bytes = my::crypto::JAMCRC_4x8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_4x8bytes == JAMCRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_16
    const auto &&jam_16bytes = my::crypto::JAMCRC_16bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_16bytes == JAMCRC_VALUE);
    const auto &&jam_16bytes_prefetch = my::crypto::JAMCRC_16bytes_prefetch(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_16bytes_prefetch == JAMCRC_VALUE);
#endif
}

BOOST_AUTO_TEST_CASE(test_crypto_CRC_4)
{
    const std::string str = "+-*/$~#[](){}&=/:;?;.!";
    const uint32_t CRC_VALUE = 0xEAFC7C67;
    // const uint32_t JAMCRC_VALUE = 0x5038398;

    const auto &&crc_boost = my::crypto::CRC32_Boost(str);
    BOOST_REQUIRE(crc_boost == CRC_VALUE);
    const auto &&crc_stackoverflow = my::crypto::CRC32_StackOverflow((const unsigned char *)str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_stackoverflow == CRC_VALUE);
    const auto &&crc_1byte_tableless = my::crypto::CRC32_1byte_tableless(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_1byte_tableless == CRC_VALUE);
    const auto &&crc_1byte_tableless2 = my::crypto::CRC32_1byte_tableless(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_1byte_tableless2 == CRC_VALUE);
    const auto &&crc_1byte = my::crypto::CRC32_1byte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_1byte == CRC_VALUE);
    const auto &&crc_bitwise = my::crypto::CRC32_bitwise(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_bitwise == CRC_VALUE);
    const auto &&crc_halfbyte = my::crypto::CRC32_halfbyte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_halfbyte == CRC_VALUE);
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_8
    const auto &&crc_4bytes = my::crypto::CRC32_4bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_4bytes == CRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_8
    const auto &&crc_8bytes = my::crypto::CRC32_8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_8bytes == CRC_VALUE);
    const auto &&crc_4x8bytes = my::crypto::CRC32_4x8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_4x8bytes == CRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_16
    const auto &&crc_16bytes = my::crypto::CRC32_16bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_16bytes == CRC_VALUE);
    const auto &&crc_16bytes_prefetch = my::crypto::CRC32_16bytes_prefetch(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(crc_16bytes_prefetch == CRC_VALUE);
#endif
    /*
    const auto &&jam_boost = my::crypto::JAMCRC_Boost(str);
    BOOST_REQUIRE(jam_boost == JAMCRC_VALUE);
    const auto &&jam_stackoverflow = my::crypto::JAMCRC_StackOverflow((const unsigned char*)str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_stackoverflow == JAMCRC_VALUE);
    const auto &&jam_1byte_tableless = my::crypto::JAMCRC_1byte_tableless(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_1byte_tableless == JAMCRC_VALUE);
    const auto &&jam_1byte_tableless2 = my::crypto::JAMCRC_1byte_tableless2(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_1byte_tableless2 == JAMCRC_VALUE);
    const auto &&jam_1byte = my::crypto::JAMCRC_1byte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_1byte == JAMCRC_VALUE);
    const auto &&jam_bitwise = my::crypto::JAMCRC_bitwise(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_bitwise == JAMCRC_VALUE);
    const auto &&jam_halfbyte = my::crypto::JAMCRC_halfbyte(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_halfbyte == JAMCRC_VALUE);
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_4
    const auto &&jam_4bytes = my::crypto::JAMCRC_4bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_4bytes == JAMCRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_8
    const auto &&jam_8bytes = my::crypto::JAMCRC_8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_8bytes == JAMCRC_VALUE);
    const auto &&jam_4x8bytes= my::crypto::JAMCRC_4x8bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_4x8bytes == JAMCRC_VALUE);
#endif
#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_16
    const auto &&jam_16bytes = my::crypto::JAMCRC_16bytes(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_16bytes == JAMCRC_VALUE);
    const auto &&jam_16bytes_prefetch = my::crypto::JAMCRC_16bytes_prefetch(str.c_str(), str.length(), 0);
    BOOST_REQUIRE(jam_16bytes_prefetch == JAMCRC_VALUE);
#endif*/
}

BOOST_AUTO_TEST_CASE(test_crypto_CRC_5)
{
    const std::string str = "Where is Brian? Brian is in the kitchen";
    const auto &&crc = my::crypto::CRC32_Boost(str);
    BOOST_REQUIRE(crc == 0x60ECAB66);
    // const auto &&jam = my::crypto::JAMCRC_Boost(str);
    // BOOST_REQUIRE((~jam) == -0x60ecab67);
}