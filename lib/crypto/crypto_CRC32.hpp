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
//  Created: 04, March, 2021                                //
//  Modified: 04, March, 2021                               //
//  file: crypto.cpp                                        //
//  Crypto                                                  //
//  Source: http://stackoverflow.com/questions/8710719/generating-an-alphabetic-sequence-in-java                                                //
//          https://github.com/theo546/stuff                                                //
//          https://stackoverflow.com/a/19299611/10152334                                                //
//          https://gms.tf/stdfind-and-memchr-optimizations.html                                                //
//          https://medium.com/applied/applied-c-align-array-elements-32af40a768ee                                                //
//          https://create.stephan-brumme.com/crc32/                                                //
//          https://rosettacode.org/wiki/Generate_lower_case_ASCII_alphabet                                                //
//          https://www.codeproject.com/Articles/663443/Cplusplus-is-Fun-Optimal-Alphabetical-Order                                                //
//          https://cppsecrets.com/users/960210997110103971089710711510497116484964103109971051084699111109/Given-integer-n-find-the-nth-string-in-this-sequence-A-B-C-Z-AA-AB-AC-ZZ-AAA-AAB-AAZ-ABA-.php
//          https://www.careercup.com/question?id=14276663                                                //
//          https://stackoverflow.com/a/55074804/10152334                                                //
//          https://stackoverflow.com/questions/26429360/crc32-vs-crc32c                                                //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

/** @file ALL CRC32 and JAMCRC function */

#ifndef CRYPTO_CRC32_HPP_
#define CRYPTO_CRC32_HPP_

#include <boost/crc.hpp>

/* CRC-32C (iSCSI) polynomial in reversed bit order. */
//#define POLY 0x82f63b78
/* CRC-32 (Ethernet, ZIP, etc.) polynomial in reversed bit order. */
#define POLY 0xEDB88320

#define CRC32_USE_LOOKUP_TABLE_BYTE
#define CRC32_USE_LOOKUP_TABLE_SLICING_BY_4
#define CRC32_USE_LOOKUP_TABLE_SLICING_BY_8
#define CRC32_USE_LOOKUP_TABLE_SLICING_BY_16

namespace my
{
namespace crypto
{

/**
 * This function process CRC32 hash from string
 *
 * @param my_string String text need to be calculate
 * @return uint32_t with CRC32 value
 * @see see also JAMCRC_Boost()
 */
uint32_t CRC32_Boost(std::string_view my_string);

/**
 * This function process JAMCRC hash from string
 *
 * @param String my_string text to process
 * @return uint32_t with JAMCRC value
 * @see see also CRC32_Boost()
 */
uint32_t JAMCRC_Boost(std::string_view my_string);

/**
 * This function process CRC32 hash from void * const
 *
 * @param buf void* text need to be calculate
 * @param len < b>size_t< /b> text size
 * @param crc < b>uint32_t< /b> least crc (0x0 if is empty)
 * @return < b>uint32_t< /b> with CRC32 value
 * @see see also JAMCRC_Boost()
 */
uint32_t CRC32_Boost(const void *buf, size_t len, uint32_t crc);
uint32_t JAMCRC_Boost(const void *buf, size_t len, uint32_t crc);

uint32_t CRC32_StackOverflow(const void *buf, size_t len, uint32_t crc);
uint32_t JAMCRC_StackOverflow(const void *buf, size_t len, uint32_t crc);

uint32_t CRC32_1byte_tableless(const void *data, size_t length, uint32_t previousCrc32);
uint32_t JAMCRC_1byte_tableless(const void *data, size_t length, uint32_t previousCrc32);

uint32_t CRC32_1byte_tableless2(const void *data, size_t length, uint32_t previousCrc32);
uint32_t JAMCRC_1byte_tableless2(const void *data, size_t length, uint32_t previousCrc32);

uint32_t CRC32_bitwise(const void *data, size_t length, uint32_t previousCrc32);
uint32_t JAMCRC_bitwise(const void *data, size_t length, uint32_t previousCrc32);

uint32_t CRC32_halfbyte(const void *data, size_t length, uint32_t previousCrc32);
uint32_t JAMCRC_halfbyte(const void *data, size_t length, uint32_t previousCrc32);

#ifdef CRC32_USE_LOOKUP_TABLE_BYTE
uint32_t CRC32_1byte(const void *data, size_t length, uint32_t previousCrc32);
uint32_t JAMCRC_1byte(const void *data, size_t length, uint32_t previousCrc32);
#endif

#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_4
uint32_t CRC32_4bytes(const void *data, size_t length, uint32_t previousCrc32 = 0);
uint32_t JAMCRC_4bytes(const void *data, size_t length, uint32_t previousCrc32 = 0);
#endif

#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_8
uint32_t CRC32_8bytes(const void *data, size_t length, uint32_t previousCrc32 = 0);
uint32_t CRC32_4x8bytes(const void *data, size_t length, uint32_t previousCrc32 = 0);
uint32_t JAMCRC_8bytes(const void *data, size_t length, uint32_t previousCrc32 = 0);
uint32_t JAMCRC_4x8bytes(const void *data, size_t length, uint32_t previousCrc32 = 0);
#endif

#ifdef CRC32_USE_LOOKUP_TABLE_SLICING_BY_16
uint32_t CRC32_16bytes(const void *data, size_t length, uint32_t previousCrc32 = 0);
uint32_t CRC32_16bytes_prefetch(const void *data, size_t length, uint32_t previousCrc32 = 0, size_t prefetchAhead = 256);
uint32_t JAMCRC_16bytes(const void *data, size_t length, uint32_t previousCrc32 = 0);
uint32_t JAMCRC_16bytes_prefetch(const void *data, size_t length, uint32_t previousCrc32 = 0, size_t prefetchAhead = 256);
#endif
uint32_t CRC32_fast(const void *data, size_t length, uint32_t previousCrc32);
uint32_t JAMCRC_fast(const void *data, size_t length, uint32_t previousCrc32);

} // namespace crypto
} // namespace my

#endif
