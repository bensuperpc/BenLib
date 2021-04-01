//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  BenLib, 2021                                            //
//  Created: 10, January, 2021                              //
//  Modified: 11, January, 2021                             //
//  file: parse_csv.cpp                                     //
//  Parse_csv                                               //
//  Source: -                                               //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////
#include <iostream>
#include <string>
#include <vector>
#include "string_lib/parsing.hpp"

/**
 * @brief 
 * 
 * @example parse_csv.cpp
 * @param argc 
 * @param argv 
 * @param envp 
 * @return int 
 */

#define PROJECT_NAME "@PROJECT_NAME@"
#define PROJECT_VER "@PROJECT_VERSION@"
#define PROJECT_VER_MAJOR "@PROJECT_VERSION_MAJOR@"
#define PROJECT_VER_MINOR "@PROJECT_VERSION_MINOR@"
#define PTOJECT_VER_PATCH "@PROJECT_VERSION_PATCH@"

int main(int argc, char *argv[], char *envp[])
{
    //std::cout << "project name: " << PROJECT_NAME << " version: " << PROJECT_VER << std::endl;
    if (argc >= 2) {
        std::vector<std::vector<std::string>> file;
        // Open CSV file and parse it
        my::string::csv_parse(file, std::string(argv[1]), ',');

        for (const auto &line : file) {
            for (const auto &element : line) {
                std::cout << element << ", ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "You must provide 1 or more arguments" << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}