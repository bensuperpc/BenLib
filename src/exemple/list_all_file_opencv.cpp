#include <vector>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "../../lib/filesystem/filesystem.hpp"

int main()
{
    std::vector<std::string> list_files = {};
    list_files.reserve(1000);

    my::filesystem::search_by_ext(list_files, ".", ".png");

    std::ios_base::sync_with_stdio(false);
    auto && image = cv::Mat();
    
    for(const auto & elem : list_files)
    {
        image = cv::imread(elem);
        if((image.cols == 1280 && image.rows == 768) || (image.cols == 1280 && image.rows == 1024)) {
        }
    }
}
