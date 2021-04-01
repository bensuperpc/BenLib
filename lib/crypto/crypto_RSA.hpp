/**
 * @file crypto_RSA.hpp
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

#ifndef CRYPTO_RSA_HPP_
#define CRYPTO_RSA_HPP_

#define BUFFSIZE 16384

// For RSA
extern "C"
{
#include <openssl/engine.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>
}
#define KEY_LENGTH 4096
#define PUBLIC_EXPONENT 65537 // 59 // Public exponent should be a prime number.
#define PUBLIC_KEY_PEM 1
#define PRIVATE_KEY_PEM 0

#define LOG(x) cout << x << endl;

//#include "openssl_rsa.h"

#define BUFFSIZE 16384

namespace my
{
namespace crypto
{
/**
 * @brief 
 * 
 * @ingroup Crypto_RSA
 *
 * @param keypair 
 * @param pem_type 
 * @param file_name 
 * @return RSA* 
 */
RSA *create_RSA(RSA *keypair, int pem_type, char *file_name);

/**
 * @brief 
 * 
 * @ingroup Crypto_RSA
 *
 * @param flen 
 * @param from 
 * @param to 
 * @param key 
 * @param padding 
 * @return int 
 */
int public_encrypt(int flen, unsigned char *from, unsigned char *to, RSA *key, int padding);

/**
 * @brief 
 * 
 * @ingroup Crypto_RSA
 *
 * @param flen 
 * @param from 
 * @param to 
 * @param key 
 * @param padding 
 * @return int 
 */
int private_decrypt(int flen, unsigned char *from, unsigned char *to, RSA *key, int padding);

/**
 * @brief Create a encrypted file object
 * 
 * @ingroup Crypto_RSA
 *
 * @param encrypted 
 * @param key_pair 
 * @param filename 
 */
void create_encrypted_file(char *encrypted, RSA *key_pair, char *filename);

} // namespace crypto
} // namespace my
// https://www.quora.com/How-can-I-get-the-MD5-or-SHA-hash-of-a-file-in-C

#endif
