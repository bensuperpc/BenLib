/**
 * @file crypto_AES.hpp
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
https://stackoverflow.com/a/5580881/10152334
*/

#ifndef CRYPTO_AES_HPP_
#define CRYPTO_AES_HPP_

// For AES
extern "C"
{
#include <openssl/aes.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <string.h>
}

#define BUFFSIZE 16384

namespace my
{
namespace crypto
{

/**
 * @brief 
 *
 * @ingroup Crypto_AES
 *
 * @param ciphertext 
 * @param ciphertext_len 
 * @param aad 
 * @param aad_len 
 * @param tag 
 * @param key 
 * @param iv 
 * @param plaintext 
 * @return int 
 */
int Decrypt_AES(unsigned char *ciphertext, int ciphertext_len, unsigned char *aad, int aad_len, unsigned char *tag, unsigned char *key, unsigned char *iv,
    unsigned char *plaintext);


/**
 * @brief 
 *
 * @ingroup Crypto_AES
 *
 * @param plaintext 
 * @param plaintext_len 
 * @param aad 
 * @param aad_len 
 * @param key 
 * @param iv 
 * @param ciphertext 
 * @param tag 
 * @return int 
 */
int Encrypt_AES(unsigned char *plaintext, int plaintext_len, unsigned char *aad, int aad_len, unsigned char *key, unsigned char *iv, unsigned char *ciphertext,
    unsigned char *tag);

/**
 * @brief 
 *
 * @ingroup Crypto_AES
 *
 * @param key 
 * @return int 
 */
int Rand_Key_AES(unsigned char *key);

/**
 * @brief 
 *
 * @ingroup Crypto_AES
 *
 * @param iv 
 * @return int 
 */
int Rand_IV_AES(unsigned char *iv);

/**
 * @brief 
 *
 * @ingroup Crypto_AES
 *
 */
__attribute__((__noreturn__)) void handleErrors();

} // namespace crypto
} // namespace my
// https://www.quora.com/How-can-I-get-the-MD5-or-SHA-hash-of-a-file-in-C

#endif
