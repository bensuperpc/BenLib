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
//  Created: 4, Novemver, 2020                              //
//  Modified: 4, Novemver, 2020                             //
//  file: image_diff.cpp                                    //
//  Benchmark CPU with Optimization                         //
//  Source: -                                               //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "utils/opencv/opencv_utils.hpp"

int main(int argc, char *argv[], char *envp[])
{
    cv::Mat &&img1 = cv::imread(argv[1], 1);
    cv::Mat &&img2 = cv::imread(argv[2], 1);

    // cv::Mat imgdiff(img1.rows, img1.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat &&imgdiff = cv::Mat();
    int th = 10;
    opencv_utils::imgdiff_prev(img1, img2, imgdiff, th);

    cv::imwrite("diff.png", imgdiff);
    return EXIT_SUCCESS;
}
