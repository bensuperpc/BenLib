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
//  Created: 4, November, 2020                              //
//  Modified: 8, November, 2020                             //
//  file: list_all_file_hash.cpp                            //
//  List all file with hash                                 //
//  Source: -                                               //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>
#include "filesystem/filesystem.hpp"

// https://bastian.rieck.me/blog/posts/2017/simple_unit_tests/


/**
 * @brief 
 * 
 * @example count_all_file.cpp
 * @param argc 
 * @param argv 
 * @param envp 
 * @return int 
 */
int main(int argc, char *argv[], char *envp[])
{
    // Return files count
    if (argc == 2) {
        std::cout << "There is " << my::filesystem::count_files(".") << " Files" << std::endl;
    } else {
        std::cout << "There is " << my::filesystem::count_files(argv[1]) << " Files" << std::endl;
    }
    return EXIT_SUCCESS;
}