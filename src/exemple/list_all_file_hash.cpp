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
//  Modified: 22, March, 2021                               //
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

/**
 * @brief 
 * 
 * @example list_all_file_hash.cpp
 * @param argc 
 * @param argv 
 * @param envp 
 * @return int 
 */
int main(int argc, char *argv[], char *envp[])
{
    if (argc < 2) {
        std::cout << "You need enter more arguments: './list_all_file_hash *path* for exemple" << std::endl;
        return EXIT_FAILURE;
    }

    // Vector to save files list
    std::vector<std::string> files;
    files.reserve(100);

    for (size_t i = 1; i < (size_t)argc; i++) {
        const std::filesystem::path path = argv[i];
        for (const auto &p : std::filesystem::recursive_directory_iterator(path)) {
            if (!std::filesystem::is_directory(p)) {
                files.emplace_back(p.path().string());
            }
        }
    }

    const std::vector<std::pair<const std::string, std::string (*)(const std::string &)>> pointer_map {{"get_md5hash", &my::crypto::get_md5hash},
        {"get_sha1hash", &my::crypto::get_sha1hash}, {"get_sha256hash", &my::crypto::get_sha256hash}, {"get_sha512hash", &my::crypto::get_sha512hash}};
#pragma omp parallel for collapse(2) ordered schedule(auto)
    for (size_t i = 0; i < files.size(); i++) {
        for (size_t j = 0; j < pointer_map.size(); j++) {
            const auto str = (pointer_map[j].second)(files[i]);
#pragma omp ordered
            std::cout << files[i] << ":" << '\n' << pointer_map[j].first << ": " << str << std::endl;
        }
    }
    return EXIT_SUCCESS;
}