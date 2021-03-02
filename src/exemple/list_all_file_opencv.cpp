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

int main()
{
    std::vector<std::string> list_files = {};
    list_files.reserve(1000);

    my::filesystem::search_by_ext(list_files, ".", ".png");

    std::ios_base::sync_with_stdio(false);
    auto &&image = cv::Mat();

    for (const auto &elem : list_files) {
        image = cv::imread(elem);
        if ((image.cols == 1280 && image.rows == 768) || (image.cols == 1280 && image.rows == 1024)) {
        }
    }
    return EXIT_SUCCESS;
}
