/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** main.hpp
*/

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
    return 0;
}
