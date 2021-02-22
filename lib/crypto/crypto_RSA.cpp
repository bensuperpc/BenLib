/*
** BENSUPERPC PROJECT, 2021
** Crypto
** File description:
** crypto.cpp
*/

// Source https://github.com/bavlayan/Encrypt-Decrypt-with-OpenSSL---RSA

#include "crypto_RSA.hpp"

RSA * my::crypto::create_RSA(RSA * keypair, int pem_type, char *file_name) {

    RSA   *rsa = NULL;
    FILE  *fp  = NULL;

    if(pem_type == PUBLIC_KEY_PEM) {

        fp = fopen(file_name, "w");
        PEM_write_RSAPublicKey(fp, keypair);
        fclose(fp);

        fp = fopen(file_name, "rb");
        PEM_read_RSAPublicKey(fp, &rsa, NULL, NULL);
        fclose(fp);

    }
    else if(pem_type == PRIVATE_KEY_PEM) {

        fp = fopen(file_name, "w");
        PEM_write_RSAPrivateKey(fp, keypair, NULL, NULL, NULL, NULL, NULL);
        fclose(fp);

        fp = fopen(file_name, "rb");
        PEM_read_RSAPrivateKey(fp, &rsa, NULL, NULL);
        fclose(fp);

    }

    return rsa;
}

int my::crypto::public_encrypt(int flen, unsigned char* from, unsigned char* to, RSA* key, int padding) {
    
    int result = RSA_public_encrypt(flen, from, to, key, padding);
    return result;
}

int my::crypto::private_decrypt(int flen, unsigned char* from, unsigned char* to, RSA* key, int padding) {

    int result = RSA_private_decrypt(flen, from, to, key, padding);
    return result;
}

void my::crypto::create_encrypted_file(char* encrypted, RSA* key_pair) {

    FILE* encrypted_file = fopen("encrypted_file.bin", "w");
    fwrite(encrypted, sizeof(*encrypted), RSA_size(key_pair), encrypted_file);
    fclose(encrypted_file);
}