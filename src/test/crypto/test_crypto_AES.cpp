#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE crypto_AES

#include <algorithm>
#include <boost/predef.h>

#include <boost/test/unit_test.hpp>
#include "crypto/crypto_AES.hpp"

#include <openssl/aes.h>
#include <openssl/err.h>
#include <openssl/evp.h>

BOOST_AUTO_TEST_CASE(test_crypto_aes_1)
{
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();

    unsigned char key[32];
    BOOST_REQUIRE(my::crypto::Rand_Key_AES(key) == 0);
    //BOOST_CHECK_EQUAL(1, arg1);

    unsigned char iv[16];
    BOOST_REQUIRE(my::crypto::Rand_IV_AES(iv) == 0);

    unsigned char plaintext[] = "The quick brown fox jumps over the lazy dog";
    unsigned char aad[] = "Some AAD data";

    unsigned char ciphertext[128];
    unsigned char decryptedtext[128];

    unsigned char tag[16];

    int decryptedtext_len = 0, ciphertext_len = 0;
    ciphertext_len
        = my::crypto::Encrypt_AES(plaintext, strlen(reinterpret_cast<char *>(plaintext)), aad, strlen(reinterpret_cast<char *>(aad)), key, iv, ciphertext, tag);

    decryptedtext_len = my::crypto::Decrypt_AES(ciphertext, ciphertext_len, aad, strlen(reinterpret_cast<char *>(aad)), tag, key, iv, decryptedtext);

    BOOST_REQUIRE(decryptedtext_len >= 0);
    decryptedtext[decryptedtext_len] = '\0';
    
    BOOST_CHECK(strlen(reinterpret_cast<char *>(decryptedtext)) == strlen(reinterpret_cast<char *>(plaintext)));

    //BOOST_REQUIRE(decryptedtext == plaintext);
    //std::cout << strlen(reinterpret_cast<char *>(decryptedtext) << " " << strlen(reinterpret_cast<char *>(plaintext) << std::endl;
    

    ERR_free_strings();
}
