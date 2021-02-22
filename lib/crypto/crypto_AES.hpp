/*
** BENSUPERPC PROJECT, 2020
** Crypto
** Source: https://stackoverflow.com/questions/178265/what-is-the-most-hard-to-understand-piece-of-c-code-you-know https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
https://github.com/bavlayan/Encrypt-Decrypt-with-OpenSSL---RSA

** crypto.cpp
*/

#ifndef CRYPTO_AES_HPP_
#define CRYPTO_AES_HPP_

// For AES
#include <openssl/aes.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <string.h>

#define BUFFSIZE 16384

namespace my
{
namespace crypto
{
int Decrypt_AES(unsigned char *ciphertext, int ciphertext_len, unsigned char *aad, int aad_len, unsigned char *tag, unsigned char *key, unsigned char *iv,
    unsigned char *plaintext);

int Encrypt_AES(unsigned char *plaintext, int plaintext_len, unsigned char *aad, int aad_len, unsigned char *key, unsigned char *iv, unsigned char *ciphertext,
    unsigned char *tag);

__attribute__((__noreturn__)) void handleErrors();

} // namespace crypto
} // namespace my
// https://www.quora.com/How-can-I-get-the-MD5-or-SHA-hash-of-a-file-in-C

#endif
