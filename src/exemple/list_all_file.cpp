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
//  Created: 20, October, 2020                              //
//  Modified: 20, October, 2020                             //
//  file: list_all_file.cpp                                 //
//  List all file                                           //
//  Source: -                                               //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////
#include <iostream>
#include <string>
#include <vector>
#include "filesystem/filesystem.hpp"

/**
 * @brief 
 * 
 * @example list_all_file.cpp
 * @param argc 
 * @param argv 
 * @param envp 
 * @return int 
 */
int main(int argc, char *argv[], char *envp[])
{
    std::vector<std::string> list_files = {};
    list_files.reserve(1000);

    my::filesystem::list_all_files(list_files, ".");

    std::ios_base::sync_with_stdio(false);
    for (const auto &elem : list_files) {
        std::cout << elem << std::endl;
    }
    std::cout << "There is " << list_files.size() << " Files" << std::endl;
    return EXIT_SUCCESS;
}