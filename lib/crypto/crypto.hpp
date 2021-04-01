/**
 * @file crypto.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

/*
** Source: https://stackoverflow.com/questions/178265/what-is-the-most-hard-to-understand-piece-of-c-code-you-know https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
https://github.com/bavlayan/Encrypt-Decrypt-with-OpenSSL---RSA
*/

#ifndef CRYPTO_HPP_
#define CRYPTO_HPP_

#define BUFFSIZE 16384

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// For MD5 and SHA
#include <boost/crc.hpp> //CRC32
#include <openssl/md5.h>
#include <openssl/sha.h>
#include <string_view>

//#include "openssl_rsa.h"

#define BUFFSIZE 16384

namespace my
{
namespace crypto
{
/**
 * @brief Get the md5hash object
 * 
 * @param str 
 * @return std::string 
 */
std::string get_md5hash(const std::string &str);
// void get_md5hash_from_string(const unsigned char *, size_t &);
/**
 * @brief Get the md5hash from string object
 * 
 * @param str 
 * @return std::string 
 */
std::string get_md5hash_from_string(const std::string &str);

/**
 * @brief Get the sha1hash object
 * 
 * @param str 
 * @return std::string 
 */
std::string get_sha1hash(const std::string &str);
// void get_sha1hash_from_string(const unsigned char *, size_t &);

/**
 * @brief Get the sha1hash from string object
 * 
 * @param str 
 * @return std::string 
 */
std::string get_sha1hash_from_string(const std::string &str);

/**
 * @brief Get the sha224hash object
 * 
 * @param str 
 * @return std::string 
 */
std::string get_sha224hash(const std::string &str);
// void get_sha224hash_from_string(const unsigned char *, size_t &);

/**
 * @brief Get the sha224hash from string object
 * 
 * @param str 
 * @return std::string 
 */
std::string get_sha224hash_from_string(const std::string &str);

/**
 * @brief Get the sha256hash object
 * 
 * @param str 
 * @return std::string 
 */
std::string get_sha256hash(const std::string &str);
// void get_sha256hash_from_string(const unsigned char *, size_t &);

/**
 * @brief Get the sha256hash from string object
 * 
 * @param str 
 * @return std::string 
 */
std::string get_sha256hash_from_string(const std::string &str);

/**
 * @brief Get the sha384hash object
 * 
 * @param str 
 * @return std::string 
 */
std::string get_sha384hash(const std::string &str);
// void get_sha384hash_from_string(const unsigned char *, size_t &);

/**
 * @brief Get the sha384hash from string object
 * 
 * @param str 
 * @return std::string 
 */
std::string get_sha384hash_from_string(const std::string &str);

/**
 * @brief Get the sha512hash object
 * 
 * @param str 
 * @return std::string 
 */
std::string get_sha512hash(const std::string &str);
// void get_sha512hash_from_string(const unsigned char *, size_t &);
/**
 * @brief Get the sha512hash from string object
 * 
 * @param str 
 * @return std::string 
 */
std::string get_sha512hash_from_string(const std::string &str);
} // namespace crypto
} // namespace my
// https://www.quora.com/How-can-I-get-the-MD5-or-SHA-hash-of-a-file-in-C

#endif
