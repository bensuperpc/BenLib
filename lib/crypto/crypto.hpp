/*
** BENSUPERPC PROJECT, 2020
** Crypto
** Source: https://stackoverflow.com/questions/178265/what-is-the-most-hard-to-understand-piece-of-c-code-you-know https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
** crypto.cpp
*/

#ifndef CRYPTO_HPP_
#define CRYPTO_HPP_

#define BUFFSIZE 16384

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/md5.h>
#include <openssl/sha.h>
#include <sstream>
#include <string>

#define BUFFSIZE 16384

namespace my
{
namespace crypto
{
// MD5
std::string get_md5hash(const std::string &);
unsigned char *get_md5hash_from_string(const unsigned char *, size_t &);
std::string get_md5hash_from_string(const std::string &str);

// SHA1
std::string get_sha1hash(const std::string &);
unsigned char *get_sha1hash_from_string(const unsigned char *, size_t &);
std::string get_sha1hash_from_string(const std::string &str);

// SHA224
std::string get_sha224hash(const std::string &);
unsigned char *get_sha224hash_from_string(const unsigned char *, size_t &);
std::string get_sha224hash_from_string(const std::string &str);

// SHA256
std::string get_sha256hash(const std::string &);
unsigned char *get_sha256hash_from_string(const unsigned char *, size_t &);
std::string get_sha256hash_from_string(const std::string &str);

// SHA384
std::string get_sha384hash(const std::string &);
unsigned char *get_sha384hash_from_string(const unsigned char *, size_t &);
std::string get_sha384hash_from_string(const std::string &str);

// SHA512
std::string get_sha512hash(const std::string &);
unsigned char *get_sha512hash_from_string(const unsigned char *, size_t &);
std::string get_sha512hash_from_string(const std::string &str);

} // namespace crypto
} // namespace my
// https://www.quora.com/How-can-I-get-the-MD5-or-SHA-hash-of-a-file-in-C

#endif
