/*
** BENSUPERPC PROJECT, 2020
** Crypto
** File description:
** crypto.cpp
*/

// Source https://www.quora.com/How-can-I-get-the-MD5-or-SHA-hash-of-a-file-in-C
#include "crypto.hpp"

std::string my::crypto::get_md5hash(const std::string &fname)
{

    char buffer[BUFFSIZE];
    unsigned char digest[MD5_DIGEST_LENGTH]; // == 16

    std::stringstream ss;
    std::string md5string;

    std::ifstream ifs(fname, std::ifstream::binary);

    MD5_CTX md5Context;

    MD5_Init(&md5Context);

    while (ifs.good()) {
        ifs.read(buffer, BUFFSIZE);
        MD5_Update(&md5Context, buffer, static_cast<size_t>(ifs.gcount()));
    }

    ifs.close();

    auto &&res = MD5_Final(digest, &md5Context);

    if (res == 0)  // hash failed
        return {}; // or raise an exception

    // ss << std::hex << std::uppercase << std::setfill('0');
    ss << std::hex << std::setfill('0');
    for (unsigned char uc : digest)
        ss << std::setw(2) << (int)uc;

    md5string = ss.str();

    return md5string;
}

std::string my::crypto::get_md5hash_from_string(const std::string &str)
{
    unsigned char hash[MD5_DIGEST_LENGTH]; // == 16
    unsigned char *cstr = new unsigned char[str.length() + 1];
    std::strcpy((char *)cstr, str.c_str());
    auto hashed = MD5(cstr, sizeof(*cstr) - 1, hash);
    std::string s(reinterpret_cast<char const *>(hashed));
    return s;
}

unsigned char *my::crypto::get_md5hash_from_string(const unsigned char *cstr, size_t &size)
{
    unsigned char hash[MD5_DIGEST_LENGTH]; // == 16
    return MD5(cstr, sizeof(*cstr) - 1, hash);
}

std::string my::crypto::get_sha1hash(const std::string &fname)
{

    char buffer[BUFFSIZE];
    unsigned char digest[SHA_DIGEST_LENGTH];

    std::stringstream ss;
    std::string sha1string;

    std::ifstream ifs(fname, std::ifstream::binary);

    SHA_CTX sha1Context;

    SHA1_Init(&sha1Context);

    while (ifs.good()) {
        ifs.read(buffer, BUFFSIZE);
        SHA1_Update(&sha1Context, buffer, static_cast<size_t>(ifs.gcount()));
    }

    ifs.close();

    auto &&res = SHA1_Final(digest, &sha1Context);

    if (res == 0)  // hash failed
        return {}; // or raise an exception

    // ss << std::hex << std::uppercase << std::setfill('0');
    ss << std::hex << std::setfill('0');
    for (unsigned char uc : digest)
        ss << std::setw(2) << (int)uc;

    sha1string = ss.str();

    return sha1string;
}

std::string my::crypto::get_sha1hash_from_string(const std::string &str)
{
    unsigned char hash[SHA_DIGEST_LENGTH]; // == 16
    unsigned char *cstr = new unsigned char[str.length() + 1];
    std::strcpy((char *)cstr, str.c_str());
    auto hashed = SHA1(cstr, sizeof(*cstr) - 1, hash);
    std::string s(reinterpret_cast<char const *>(hashed));
    return s;
}

unsigned char *my::crypto::get_sha1hash_from_string(const unsigned char *cstr, size_t &size)
{
    unsigned char hash[SHA_DIGEST_LENGTH]; // == 16
    return SHA1(cstr, sizeof(*cstr) - 1, hash);
}

std::string my::crypto::get_sha224hash(const std::string &fname)
{

    char buffer[BUFFSIZE];
    unsigned char digest[SHA224_DIGEST_LENGTH];

    std::stringstream ss;
    std::string sha224string;

    std::ifstream ifs(fname, std::ifstream::binary);

    SHA256_CTX sha224Context;

    SHA224_Init(&sha224Context);

    while (ifs.good()) {
        ifs.read(buffer, BUFFSIZE);
        SHA256_Update(&sha224Context, buffer, static_cast<size_t>(ifs.gcount()));
    }

    ifs.close();

    auto &&res = SHA256_Final(digest, &sha224Context);

    if (res == 0)  // hash failed
        return {}; // or raise an exception

    // ss << std::hex << std::uppercase << std::setfill('0');
    ss << std::hex << std::setfill('0');
    for (unsigned char uc : digest)
        ss << std::setw(2) << (int)uc;

    sha224string = ss.str();

    return sha224string;
}

std::string my::crypto::get_sha224hash_from_string(const std::string &str)
{
    unsigned char hash[SHA224_DIGEST_LENGTH]; // == 16
    unsigned char *cstr = new unsigned char[str.length() + 1];
    std::strcpy((char *)cstr, str.c_str());
    auto hashed = SHA224(cstr, sizeof(*cstr) - 1, hash);
    std::string s(reinterpret_cast<char const *>(hashed));
    return s;
}

unsigned char *my::crypto::get_sha224hash_from_string(const unsigned char *cstr, size_t &size)
{
    unsigned char hash[SHA224_DIGEST_LENGTH]; // == 16
    return SHA224(cstr, sizeof(*cstr) - 1, hash);
}

std::string my::crypto::get_sha256hash(const std::string &fname)
{

    char buffer[BUFFSIZE];
    unsigned char digest[SHA256_DIGEST_LENGTH];

    std::stringstream ss;
    std::string sha256string;

    std::ifstream ifs(fname, std::ifstream::binary);

    SHA256_CTX sha256Context;

    SHA256_Init(&sha256Context);

    while (ifs.good()) {
        ifs.read(buffer, BUFFSIZE);
        SHA256_Update(&sha256Context, buffer, static_cast<size_t>(ifs.gcount()));
    }

    ifs.close();

    auto &&res = SHA256_Final(digest, &sha256Context);

    if (res == 0)  // hash failed
        return {}; // or raise an exception

    // ss << std::hex << std::uppercase << std::setfill('0');
    ss << std::hex << std::setfill('0');
    for (unsigned char uc : digest)
        ss << std::setw(2) << (int)uc;

    sha256string = ss.str();

    return sha256string;
}

std::string my::crypto::get_sha256hash_from_string(const std::string &str)
{
    unsigned char hash[SHA256_DIGEST_LENGTH]; // == 16
    unsigned char *cstr = new unsigned char[str.length() + 1];
    std::strcpy((char *)cstr, str.c_str());
    auto hashed = SHA256(cstr, sizeof(*cstr) - 1, hash);
    std::string s(reinterpret_cast<char const *>(hashed));
    return s;
}

unsigned char *my::crypto::get_sha256hash_from_string(const unsigned char *cstr, size_t &size)
{
    unsigned char hash[SHA256_DIGEST_LENGTH]; // == 16
    return SHA256(cstr, sizeof(*cstr) - 1, hash);
}

std::string my::crypto::get_sha384hash(const std::string &fname)
{

    char buffer[BUFFSIZE];
    unsigned char digest[SHA384_DIGEST_LENGTH];

    std::stringstream ss;
    std::string sha384string;

    std::ifstream ifs(fname, std::ifstream::binary);

    SHA512_CTX sha384Context;

    SHA384_Init(&sha384Context);

    while (ifs.good()) {
        ifs.read(buffer, BUFFSIZE);
        SHA384_Update(&sha384Context, buffer, static_cast<size_t>(ifs.gcount()));
    }

    ifs.close();

    auto &&res = SHA384_Final(digest, &sha384Context);

    if (res == 0)  // hash failed
        return {}; // or raise an exception

    // ss << std::hex << std::uppercase << std::setfill('0');
    ss << std::hex << std::setfill('0');
    for (unsigned char uc : digest)
        ss << std::setw(2) << (int)uc;

    sha384string = ss.str();

    return sha384string;
}

std::string my::crypto::get_sha384hash_from_string(const std::string &str)
{
    unsigned char hash[SHA384_DIGEST_LENGTH]; // == 16
    unsigned char *cstr = new unsigned char[str.length() + 1];
    std::strcpy((char *)cstr, str.c_str());
    auto hashed = SHA384(cstr, sizeof(*cstr) - 1, hash);
    std::string s(reinterpret_cast<char const *>(hashed));
    return s;
}

unsigned char *my::crypto::get_sha384hash_from_string(const unsigned char *cstr, size_t &size)
{
    unsigned char hash[SHA384_DIGEST_LENGTH]; // == 16
    return SHA384(cstr, sizeof(*cstr) - 1, hash);
}


std::string my::crypto::get_sha512hash(const std::string &fname)
{

    char buffer[BUFFSIZE];
    unsigned char digest[SHA512_DIGEST_LENGTH];

    std::stringstream ss;
    std::string sha512string;

    std::ifstream ifs(fname, std::ifstream::binary);

    SHA512_CTX sha512Context;

    SHA512_Init(&sha512Context);

    while (ifs.good()) {
        ifs.read(buffer, BUFFSIZE);
        SHA512_Update(&sha512Context, buffer, static_cast<size_t>(ifs.gcount()));
    }

    ifs.close();

    auto &&res = SHA512_Final(digest, &sha512Context);

    if (res == 0)  // hash failed
        return {}; // or raise an exception

    // ss << std::hex << std::uppercase << std::setfill('0');
    ss << std::hex << std::setfill('0');
    for (unsigned char uc : digest)
        ss << std::setw(2) << (int)uc;

    sha512string = ss.str();

    return sha512string;
}

std::string my::crypto::get_sha512hash_from_string(const std::string &str)
{
    unsigned char hash[SHA512_DIGEST_LENGTH]; // == 16
    unsigned char *cstr = new unsigned char[str.length() + 1];
    std::strcpy((char *)cstr, str.c_str());
    auto hashed = SHA512(cstr, sizeof(*cstr) - 1, hash);
    std::string s(reinterpret_cast<char const *>(hashed));
    return s;
}

unsigned char *my::crypto::get_sha512hash_from_string(const unsigned char *cstr, size_t &size)
{
    unsigned char hash[SHA512_DIGEST_LENGTH]; // == 16
    return SHA512(cstr, sizeof(*cstr) - 1, hash);
}
