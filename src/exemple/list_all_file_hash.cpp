#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/md5.h>
#include <openssl/sha.h>
#include <sstream>
#include <string>
#include <vector>
#include "../../lib/filesystem/filesystem.hpp"
#if __cplusplus >= 201703L
#    include "../../lib/thread/Pool.hpp"
#else
#    include "../../lib/thread/ThreadPool.h"
#endif

#define BUFFSIZE 16384

struct Processor
{
    std::string operator()(std::function<std::string(const std::string &)> elem_fn, const std::string &elem)
    {
        return (elem_fn)(elem);
    }
};

// Source https://www.quora.com/How-can-I-get-the-MD5-or-SHA-hash-of-a-file-in-C

std::string get_md5hash(const std::string &fname)
{

    char buffer[BUFFSIZE];
    unsigned char digest[MD5_DIGEST_LENGTH];

    std::stringstream ss;
    std::string md5string;

    std::ifstream ifs(fname, std::ifstream::binary);

    MD5_CTX md5Context;

    MD5_Init(&md5Context);

    while (ifs.good()) {
        ifs.read(buffer, BUFFSIZE);
        MD5_Update(&md5Context, buffer, ifs.gcount());
    }

    ifs.close();

    int res = MD5_Final(digest, &md5Context);

    if (res == 0)  // hash failed
        return {}; // or raise an exception

    // ss << std::hex << std::uppercase << std::setfill('0');
    ss << std::hex << std::setfill('0');
    for (unsigned char uc : digest)
        ss << std::setw(2) << (int)uc;

    md5string = ss.str();

    return md5string;
}

std::string get_sha1hash(const std::string &fname)
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
        SHA1_Update(&sha1Context, buffer, ifs.gcount());
    }

    ifs.close();

    int res = SHA1_Final(digest, &sha1Context);

    if (res == 0)  // hash failed
        return {}; // or raise an exception

    // ss << std::hex << std::uppercase << std::setfill('0');
    ss << std::hex << std::setfill('0');
    for (unsigned char uc : digest)
        ss << std::setw(2) << (int)uc;

    sha1string = ss.str();

    return sha1string;
}

std::string get_sha256hash(const std::string &fname)
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
        SHA256_Update(&sha256Context, buffer, ifs.gcount());
    }

    ifs.close();

    int res = SHA256_Final(digest, &sha256Context);

    if (res == 0)  // hash failed
        return {}; // or raise an exception

    // ss << std::hex << std::uppercase << std::setfill('0');
    ss << std::hex << std::setfill('0');
    for (unsigned char uc : digest)
        ss << std::setw(2) << (int)uc;

    sha256string = ss.str();

    return sha256string;
}

std::string get_sha512hash(const std::string &fname)
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
        SHA512_Update(&sha512Context, buffer, ifs.gcount());
    }

    ifs.close();

    int res = SHA512_Final(digest, &sha512Context);

    if (res == 0)  // hash failed
        return {}; // or raise an exception

    // ss << std::hex << std::uppercase << std::setfill('0');
    ss << std::hex << std::setfill('0');
    for (unsigned char uc : digest)
        ss << std::setw(2) << (int)uc;

    sha512string = ss.str();

    return sha512string;
}

int main()
{
    std::vector<std::string> list_files = {};
    list_files.reserve(1000);

    my::filesystem::list_all_files(list_files, ".");

    std::ios_base::sync_with_stdio(false);

    std::vector<std::pair<std::string, std::vector<std::future<std::string>>>> results {};
#if __cplusplus >= 201703L
    thread::Pool thread_pool(8);
#else
    ThreadPool thread_pool(8);
#endif

    const std::vector<std::pair<const std::string, std::string (*)(const std::string &)>> pointer_map {
        {"get_md5hash", &get_md5hash}, {"get_sha1hash", &get_sha1hash}, {"get_sha256hash", &get_sha256hash}, {"get_sha512hash", &get_sha512hash}};

    results.reserve(list_files.size());

    for (auto &elem_fn : pointer_map) {
        results.emplace_back(std::pair<std::string, std::vector<std::future<std::string>>>());
        results.back().first = elem_fn.first;
        results.back().second.reserve(list_files.size());
        for (auto &file : list_files) {
            results.back().second.emplace_back(thread_pool.enqueue(Processor(), elem_fn.second, file));
        }
    }

    size_t count = 0;
    const size_t xElem = 15;
    std::vector<std::pair<std::string, std::vector<std::string>>> time {};
    time.reserve(results.size());
    for (auto &y : results) {
        time.emplace_back(std::pair<std::string, std::vector<std::string>>());
        time.back().second.reserve(y.second.size());
        time.back().first = y.first;
        for (auto &x : y.second) {
            time.back().second.emplace_back(x.get());
            count++;
            if (count % xElem == 0) {
                std::cout << double(count) / double(list_files.size() * pointer_map.size()) * 100.0f << " %" << std::endl;
            }
        }
    }
    std::cout << "Get data: OK" << std::endl;
    /*
    for (const auto &elem : list_files) {
        std::cout << elem << std::endl;
        std::cout << "MD5 : " << get_md5hash(elem) << std::endl;
        std::cout << "SHA1 : " << get_sha1hash(elem) << std::endl;
        std::cout << "SHA256 : " << get_sha256hash(elem) << std::endl;
        std::cout << "SHA512 : " << get_sha512hash(elem) << std::endl;
    }*/
}