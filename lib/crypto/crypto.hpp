/*
** BENSUPERPC PROJECT, 2020
** Crypto
** Source: https://stackoverflow.com/questions/178265/what-is-the-most-hard-to-understand-piece-of-c-code-you-know https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
** crypto.cpp
*/

#ifndef CRYPTO_HPP_
#define CRYPTO_HPP_

#define BUFFSIZE 16384

#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/md5.h>
#include <openssl/sha.h>
#include <sstream>
#include <string>

using namespace std;

namespace my
{
namespace crypto
{
std::string get_md5hash(const std::string &);
std::string get_sha1hash(const std::string &);
std::string get_sha256hash(const std::string &);
std::string get_sha512hash(const std::string &);

} // namespace crypto
} // namespace my
// https://www.quora.com/How-can-I-get-the-MD5-or-SHA-hash-of-a-file-in-C

#endif
