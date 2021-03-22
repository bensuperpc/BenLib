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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "crypto/crypto.hpp"

int main(int argc, char *argv[], char *envp[])
{
    const std::filesystem::path path = ".";

    const std::vector<std::pair<const std::string, std::string (*)(const std::string &)>> pointer_map {{"get_md5hash", &my::crypto::get_md5hash},
        {"get_sha1hash", &my::crypto::get_sha1hash}, {"get_sha256hash", &my::crypto::get_sha256hash}, {"get_sha512hash", &my::crypto::get_sha512hash}};

    for (const auto &p : std::filesystem::recursive_directory_iterator(path)) {
        if (!std::filesystem::is_directory(p)) {
#pragma omp parallel for schedule(auto) ordered
            for (size_t i = 0; i < pointer_map.size(); i++) {
                auto str = (pointer_map[i].second)(p.path().string());
#pragma omp ordered
                std::cout << p.path().string() << ":" << '\n' << pointer_map[i].first << ": " << str << '\n';
            }
        }
    }
    return EXIT_SUCCESS;
}