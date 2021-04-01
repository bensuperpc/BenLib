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
//  Created: 23, October, 2020                              //
//  Modified: 8, November, 2020                             //
//  file: list_all_file_hash.cpp                            //
//  List all file with hash                                 //
//  Source: -                                               //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "filesystem/filesystem.hpp"

/**
 * @brief 
 * 
 * @example list_all_file_opencv.cpp
 * @param argc 
 * @param argv 
 * @param envp 
 * @return int 
 */
int main(int argc, char *argv[], char *envp[])
{
    std::vector<std::string> list_files = {};
    list_files.reserve(1000);

    // List all files with .png extension
    my::filesystem::search_by_ext(list_files, ".", ".png");

    // Improve cout speed
    std::ios_base::sync_with_stdio(false);
    auto &&image = cv::Mat();

    for (const auto &elem : list_files) {
        image = cv::imread(elem);
        if ((image.cols == 1280 && image.rows == 768) || (image.cols == 1280 && image.rows == 1024)) {
        }
    }
    return EXIT_SUCCESS;
}
