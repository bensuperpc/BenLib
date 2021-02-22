/*
** BENSUPERPC PROJECT, 2020
** Crypto
** Source: https://stackoverflow.com/questions/178265/what-is-the-most-hard-to-understand-piece-of-c-code-you-know https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
https://github.com/bavlayan/Encrypt-Decrypt-with-OpenSSL---RSA

** crypto.cpp
*/

#ifndef CRYPTO_RSA_HPP_
#define CRYPTO_RSA_HPP_

#define BUFFSIZE 16384

// For RSA
#include <openssl/engine.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>
#define KEY_LENGTH 4096
#define PUBLIC_EXPONENT 59 // Public exponent should be a prime number.
#define PUBLIC_KEY_PEM 1
#define PRIVATE_KEY_PEM 0

#define LOG(x) cout << x << endl;

//#include "openssl_rsa.h"

#define BUFFSIZE 16384

namespace my
{
namespace crypto
{
RSA *create_RSA(RSA *keypair, int pem_type, char *file_name);

int public_encrypt(int flen, unsigned char *from, unsigned char *to, RSA *key, int padding);

int private_decrypt(int flen, unsigned char *from, unsigned char *to, RSA *key, int padding);

void create_encrypted_file(char *encrypted, RSA *key_pair);

} // namespace crypto
} // namespace my
// https://www.quora.com/How-can-I-get-the-MD5-or-SHA-hash-of-a-file-in-C

#endif
