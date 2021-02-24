#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE crypto_AES

#include <algorithm>
#include <boost/predef.h>
#include <boost/test/unit_test.hpp>
#include <openssl/aes.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include "crypto/crypto_AES.hpp"

BOOST_AUTO_TEST_CASE(test_crypto_aes_1)
{
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();

    unsigned char key[32];
    BOOST_REQUIRE(my::crypto::Rand_Key_AES(key) == 0);

    unsigned char key2[32];
    BOOST_REQUIRE(my::crypto::Rand_Key_AES(key2) == 0);
    BOOST_REQUIRE(strcmp((char *)key, (char *)key2) != 0);


    unsigned char iv[16];
    BOOST_REQUIRE(my::crypto::Rand_IV_AES(iv) == 0);


    unsigned char iv2[16];
    BOOST_REQUIRE(my::crypto::Rand_IV_AES(iv2) == 0);
    BOOST_REQUIRE(strcmp((char *)iv, (char *)iv2) != 0);

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

    BOOST_REQUIRE(strlen(reinterpret_cast<char *>(decryptedtext)) == strlen(reinterpret_cast<char *>(plaintext)));
    BOOST_REQUIRE(strcmp((char *)decryptedtext, (char *)plaintext) == 0);

    ERR_free_strings();
}


BOOST_AUTO_TEST_CASE(test_crypto_aes_2)
{
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();

    unsigned char key[32];
    BOOST_REQUIRE(my::crypto::Rand_Key_AES(key) == 0);

    unsigned char key2[32];
    BOOST_REQUIRE(my::crypto::Rand_Key_AES(key2) == 0);
    BOOST_REQUIRE(strcmp((char *)key, (char *)key2) != 0);


    unsigned char iv[16];
    BOOST_REQUIRE(my::crypto::Rand_IV_AES(iv) == 0);


    unsigned char iv2[16];
    BOOST_REQUIRE(my::crypto::Rand_IV_AES(iv2) == 0);
    BOOST_REQUIRE(strcmp((char *)iv, (char *)iv2) != 0);

    unsigned char plaintext[] = "Bonjour, je suis là depuis des années, transformée en chatton";
    unsigned char aad[] = "";

    unsigned char ciphertext[128];
    unsigned char decryptedtext[128];

    unsigned char tag[16];

    int decryptedtext_len = 0, ciphertext_len = 0;
    ciphertext_len
        = my::crypto::Encrypt_AES(plaintext, strlen(reinterpret_cast<char *>(plaintext)), aad, strlen(reinterpret_cast<char *>(aad)), key, iv, ciphertext, tag);

    decryptedtext_len = my::crypto::Decrypt_AES(ciphertext, ciphertext_len, aad, strlen(reinterpret_cast<char *>(aad)), tag, key, iv, decryptedtext);

    BOOST_REQUIRE(decryptedtext_len >= 0);
    decryptedtext[decryptedtext_len] = '\0';

    BOOST_REQUIRE(strlen(reinterpret_cast<char *>(decryptedtext)) == strlen(reinterpret_cast<char *>(plaintext)));
    BOOST_REQUIRE(strcmp((char *)decryptedtext, (char *)plaintext) == 0);

    ERR_free_strings();
}


BOOST_AUTO_TEST_CASE(test_crypto_aes_3)
{
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();

    unsigned char key[32];
    BOOST_REQUIRE(my::crypto::Rand_Key_AES(key) == 0);

    unsigned char key2[32];
    BOOST_REQUIRE(my::crypto::Rand_Key_AES(key2) == 0);
    BOOST_REQUIRE(strcmp((char *)key, (char *)key2) != 0);


    unsigned char iv[16];
    BOOST_REQUIRE(my::crypto::Rand_IV_AES(iv) == 0);


    unsigned char iv2[16];
    BOOST_REQUIRE(my::crypto::Rand_IV_AES(iv2) == 0);
    BOOST_REQUIRE(strcmp((char *)iv, (char *)iv2) != 0);

    unsigned char plaintext[] = "Les chaussettes de l'archiduchesse sont-elles sèches?  Archi-sèches ?";
    unsigned char aad[] = "";

    unsigned char ciphertext[128];
    unsigned char decryptedtext[128];

    unsigned char tag[16];

    int decryptedtext_len = 0, ciphertext_len = 0;
    ciphertext_len
        = my::crypto::Encrypt_AES(plaintext, strlen(reinterpret_cast<char *>(plaintext)), aad, strlen(reinterpret_cast<char *>(aad)), key, iv, ciphertext, tag);

    decryptedtext_len = my::crypto::Decrypt_AES(ciphertext, ciphertext_len, aad, strlen(reinterpret_cast<char *>(aad)), tag, key, iv, decryptedtext);

    BOOST_REQUIRE(decryptedtext_len >= 0);
    decryptedtext[decryptedtext_len] = '\0';

    BOOST_REQUIRE(strlen(reinterpret_cast<char *>(decryptedtext)) == strlen(reinterpret_cast<char *>(plaintext)));
    BOOST_REQUIRE(strcmp((char *)decryptedtext, (char *)plaintext) == 0);

    ERR_free_strings();
}

BOOST_AUTO_TEST_CASE(test_crypto_aes_4)
{
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();

    unsigned char key[32];
    BOOST_REQUIRE(my::crypto::Rand_Key_AES(key) == 0);

    unsigned char key2[32];
    BOOST_REQUIRE(my::crypto::Rand_Key_AES(key2) == 0);
    BOOST_REQUIRE(strcmp((char *)key, (char *)key2) != 0);


    unsigned char iv[16];
    BOOST_REQUIRE(my::crypto::Rand_IV_AES(iv) == 0);


    unsigned char iv2[16];
    BOOST_REQUIRE(my::crypto::Rand_IV_AES(iv2) == 0);
    BOOST_REQUIRE(strcmp((char *)iv, (char *)iv2) != 0);

    unsigned char plaintext[] = "Un chasseur sachant chasser doit savoir chasser sans son chien.";
    unsigned char aad[] = "Cinq chiens chassent six chats.";

    unsigned char ciphertext[128];
    unsigned char decryptedtext[128];

    unsigned char tag[16];

    int decryptedtext_len = 0, ciphertext_len = 0;
    ciphertext_len
        = my::crypto::Encrypt_AES(plaintext, strlen(reinterpret_cast<char *>(plaintext)), aad, strlen(reinterpret_cast<char *>(aad)), key, iv, ciphertext, tag);

    decryptedtext_len = my::crypto::Decrypt_AES(ciphertext, ciphertext_len, aad, strlen(reinterpret_cast<char *>(aad)), tag, key, iv, decryptedtext);

    BOOST_REQUIRE(decryptedtext_len >= 0);
    decryptedtext[decryptedtext_len] = '\0';

    BOOST_REQUIRE(strlen(reinterpret_cast<char *>(decryptedtext)) == strlen(reinterpret_cast<char *>(plaintext)));
    BOOST_REQUIRE(strcmp((char *)decryptedtext, (char *)plaintext) == 0);

    ERR_free_strings();
}


BOOST_AUTO_TEST_CASE(test_crypto_aes_5)
{
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();

    unsigned char key[32];
    BOOST_REQUIRE(my::crypto::Rand_Key_AES(key) == 0);

    unsigned char key2[32];
    BOOST_REQUIRE(my::crypto::Rand_Key_AES(key2) == 0);
    BOOST_REQUIRE(strcmp((char *)key, (char *)key2) != 0);


    unsigned char iv[16];
    BOOST_REQUIRE(my::crypto::Rand_IV_AES(iv) == 0);


    unsigned char iv2[16];
    BOOST_REQUIRE(my::crypto::Rand_IV_AES(iv2) == 0);
    BOOST_REQUIRE(strcmp((char *)iv, (char *)iv2) != 0);

    unsigned char plaintext[] = "lol";
    unsigned char aad[] = "Chattons";

    unsigned char ciphertext[128];
    unsigned char decryptedtext[128];

    unsigned char tag[16];

    int decryptedtext_len = 0, ciphertext_len = 0;
    ciphertext_len
        = my::crypto::Encrypt_AES(plaintext, strlen(reinterpret_cast<char *>(plaintext)), aad, strlen(reinterpret_cast<char *>(aad)), key, iv, ciphertext, tag);

    decryptedtext_len = my::crypto::Decrypt_AES(ciphertext, ciphertext_len, aad, strlen(reinterpret_cast<char *>(aad)), tag, key, iv, decryptedtext);

    BOOST_REQUIRE(decryptedtext_len >= 0);
    decryptedtext[decryptedtext_len] = '\0';

    BOOST_REQUIRE(strlen(reinterpret_cast<char *>(decryptedtext)) == strlen(reinterpret_cast<char *>(plaintext)));
    BOOST_REQUIRE(strcmp((char *)decryptedtext, (char *)plaintext) == 0);

    ERR_free_strings();
}

BOOST_AUTO_TEST_CASE(test_crypto_aes_6)
{
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();

    unsigned char key[32];
    BOOST_REQUIRE(my::crypto::Rand_Key_AES(key) == 0);

    unsigned char key2[32];
    BOOST_REQUIRE(my::crypto::Rand_Key_AES(key2) == 0);
    BOOST_REQUIRE(strcmp((char *)key, (char *)key2) != 0);


    unsigned char iv[16];
    BOOST_REQUIRE(my::crypto::Rand_IV_AES(iv) == 0);


    unsigned char iv2[16];
    BOOST_REQUIRE(my::crypto::Rand_IV_AES(iv2) == 0);
    BOOST_REQUIRE(strcmp((char *)iv, (char *)iv2) != 0);

    unsigned char plaintext[] = "lol";
    unsigned char aad[] = "";

    unsigned char ciphertext[128];
    unsigned char decryptedtext[128];

    unsigned char tag[16];

    int decryptedtext_len = 0, ciphertext_len = 0;
    ciphertext_len
        = my::crypto::Encrypt_AES(plaintext, strlen(reinterpret_cast<char *>(plaintext)), aad, strlen(reinterpret_cast<char *>(aad)), key, iv, ciphertext, tag);

    decryptedtext_len = my::crypto::Decrypt_AES(ciphertext, ciphertext_len, aad, strlen(reinterpret_cast<char *>(aad)), tag, key, iv, decryptedtext);

    BOOST_REQUIRE(decryptedtext_len >= 0);
    decryptedtext[decryptedtext_len] = '\0';

    BOOST_REQUIRE(strlen(reinterpret_cast<char *>(decryptedtext)) == strlen(reinterpret_cast<char *>(plaintext)));
    BOOST_REQUIRE(strcmp((char *)decryptedtext, (char *)plaintext) == 0);

    ERR_free_strings();
}

BOOST_AUTO_TEST_CASE(test_crypto_aes_7)
{
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();

    unsigned char key[32];
    BOOST_REQUIRE(my::crypto::Rand_Key_AES(key) == 0);

    unsigned char key2[32];
    BOOST_REQUIRE(my::crypto::Rand_Key_AES(key2) == 0);
    BOOST_REQUIRE(strcmp((char *)key, (char *)key2) != 0);


    unsigned char iv[16];
    BOOST_REQUIRE(my::crypto::Rand_IV_AES(iv) == 0);


    unsigned char iv2[16];
    BOOST_REQUIRE(my::crypto::Rand_IV_AES(iv2) == 0);
    BOOST_REQUIRE(strcmp((char *)iv, (char *)iv2) != 0);

    unsigned char plaintext[] = "lol";
    unsigned char aad[] = "";

    unsigned char ciphertext[128];
    unsigned char decryptedtext[128];

    unsigned char tag[16];

    int decryptedtext_len = 0, ciphertext_len = 0;
    ciphertext_len
        = my::crypto::Encrypt_AES(plaintext, strlen(reinterpret_cast<char *>(plaintext)), aad, strlen(reinterpret_cast<char *>(aad)), key, iv, ciphertext, tag);

    decryptedtext_len = my::crypto::Decrypt_AES(ciphertext, ciphertext_len, aad, strlen(reinterpret_cast<char *>(aad)), tag, key, iv, decryptedtext);

    BOOST_REQUIRE(decryptedtext_len >= 0);
    decryptedtext[decryptedtext_len] = '\0';

    BOOST_REQUIRE(strlen(reinterpret_cast<char *>(decryptedtext)) == strlen(reinterpret_cast<char *>(plaintext)));
    BOOST_REQUIRE(strcmp((char *)decryptedtext, (char *)plaintext) == 0);

    ERR_free_strings();
}