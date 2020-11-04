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
//  Created: 1, November, 2020                              //
//  Modified: 1, November, 2020                             //
//  file: list_all_file_hash.cpp                            //
//  List all file with hash                                 //
//  Source: -                                               //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////
#include <string>
#include <vector>
#include "../../lib/crypto/crypto.hpp"
#include "../../lib/filesystem/filesystem.hpp"
#if __cplusplus >= 201703L
#    include "../../lib/thread/Pool.hpp"
#else
#    include "../../lib/thread/ThreadPool.h"
#endif

struct Processor
{
    std::pair<std::string, std::string> operator()(std::function<std::string(const std::string &)> elem_fn, const std::string &elem)
    {
        return std::make_pair(elem, (elem_fn)(elem));
    }
};

// Source https://www.quora.com/How-can-I-get-the-MD5-or-SHA-hash-of-a-file-in-C

int main()
{
    std::vector<std::string> list_files = {};
    list_files.reserve(1000);

    my::filesystem::list_all_files(list_files, ".");

    std::ios_base::sync_with_stdio(false);

    std::vector<std::pair<std::string, std::vector<std::future<std::pair<std::string, std::string>>>>> results {};
#if __cplusplus >= 201703L
    thread::Pool thread_pool(12);
#else
    ThreadPool thread_pool(12);
#endif

    const std::vector<std::pair<const std::string, std::string (*)(const std::string &)>> pointer_map {{"get_md5hash", &my::crypto::get_md5hash},
        {"get_sha1hash", &my::crypto::get_sha1hash}, {"get_sha256hash", &my::crypto::get_sha256hash}, {"get_sha512hash", &my::crypto::get_sha512hash}};

    results.reserve(list_files.size());

    for (auto &elem_fn : pointer_map) {
        results.emplace_back(std::pair<std::string, std::vector<std::future<std::pair<std::string, std::string>>>>());
        results.back().first = elem_fn.first;
        results.back().second.reserve(list_files.size());
        for (auto &file : list_files) {
            results.back().second.emplace_back(thread_pool.enqueue(Processor(), elem_fn.second, file));
        }
    }

    size_t count = 0;
    const size_t xElem = 15;
    std::vector<std::pair<std::string, std::vector<std::pair<std::string, std::string>>>> time {};
    time.reserve(results.size());
    for (auto &y : results) {
        time.emplace_back(std::pair<std::string, std::vector<std::pair<std::string, std::string>>>());
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
    for (const auto &x : time) {
        // std::cout << x.first << std::endl;
        for (const auto &y : x.second) {
            std::cout << x.first << " : " << y.first << " : " << y.second << std::endl;
        }
    }
}