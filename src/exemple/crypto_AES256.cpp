//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  BenLib, 2020                                            //
//  Created: 20, February, 2021                             //
//  Modified: 20, February, 2021                            //
//  file: crypto.cpp                                        //
//  Crypto                                                  //
//  Source: https://stackoverflow.com/questions/9889492/how-to-do-encryption-using-aes-in-openssl                                               //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <iomanip>
#include <iostream>
extern "C"
{
#include <openssl/aes.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <string.h>
}
#include "crypto/crypto_AES.hpp"

/**
 * @brief 
 * 
 * @example crypto_AES256.cpp
 * @param arc 
 * @param argv 
 * @return int 
 */
int main(int arc, char *argv[])
{
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();

    /* Set up the key and iv. Do I need to say to not hard code these in a real application? :-) */

    /* A 256 bit key */
    // unsigned char key[] = "01234567890123456789012345678901";

    unsigned char key[32];

    if (my::crypto::Rand_Key_AES(key)) {
        printf("Error\n");
        return 0;
    } else {
        // printf("Key is:%s\n", key);
        printf("Key is:\n");
        BIO_dump_fp(stdout, reinterpret_cast<const char *>(key), 32);
    }

    /* A 128 bit IV */
    // unsigned char iv[] = "0123456789012345";
    unsigned char iv[16];
    if (my::crypto::Rand_IV_AES(iv)) {
        printf("Error\n");
        return 0;
    } else {
        // printf("Key is:%s\n", key);
        printf("IV is:\n");
        BIO_dump_fp(stdout, reinterpret_cast<const char *>(key), 16);
    }

    /* Message to be encrypted */
    unsigned char plaintext[] = "The quick brown fox jumps over the lazy dog";

    /* Some additional data to be authenticated */
    unsigned char aad[] = "Some AAD data";

    /* Buffer for ciphertext. Ensure the buffer is long enough for the
     * ciphertext which may be longer than the plaintext, dependant on the
     * algorithm and mode
     */
    unsigned char ciphertext[128];

    /* Buffer for the decrypted text */
    unsigned char decryptedtext[128];

    /* Buffer for the tag */
    unsigned char tag[16];

    int decryptedtext_len = 0, ciphertext_len = 0;

    /* Encrypt the plaintext */
    ciphertext_len
        = my::crypto::Encrypt_AES(plaintext, strlen(reinterpret_cast<char *>(plaintext)), aad, strlen(reinterpret_cast<char *>(aad)), key, iv, ciphertext, tag);

    /*
    for (const unsigned char* p = ciphertext; *p; ++p)
    {
        printf("0x%02x, ", *p);
    }*/
    printf("\n");

    /* Do something useful with the ciphertext here */
    printf("Ciphertext is:\n");
    BIO_dump_fp(stdout, reinterpret_cast<const char *>(ciphertext), ciphertext_len);
    printf("Tag is:\n");
    BIO_dump_fp(stdout, reinterpret_cast<const char *>(tag), 14);

    /* Mess with stuff */
    /* ciphertext[0] ^= 1; */
    /* tag[0] ^= 1; */

    /* Decrypt the ciphertext */
    decryptedtext_len = my::crypto::Decrypt_AES(ciphertext, ciphertext_len, aad, strlen(reinterpret_cast<char *>(aad)), tag, key, iv, decryptedtext);

    if (decryptedtext_len < 0) {
        /* Verify error */
        printf("Decrypted text failed to verify\n");
    } else {
        /* Add a NULL terminator. We are expecting printable text */
        decryptedtext[decryptedtext_len] = '\0';

        /* Show the decrypted text */
        printf("Decrypted text is:\n");
        printf("%s\n", decryptedtext);
    }

    /* Remove error strings */
    ERR_free_strings();

    return EXIT_SUCCESS;
}