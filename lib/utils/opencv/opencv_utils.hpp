/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** opencv_utils.hpp
*/

#ifndef _OPENCV_UTILS_HPP_
#define _OPENCV_UTILS_HPP_

// Disable Warning from OpenCV libs
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#pragma GCC diagnostic ignored "-Wswitch-default"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop

namespace opencv_utils
{
cv::Mat cropimgrect(const cv::Mat &, const cv::Rect &);
void cropimgrect(const cv::Mat &, cv::Mat &, const cv::Rect &);
void imgdiff_prev(cv::Mat &, cv::Mat &, cv::Mat &);
void imgdiff_prev(cv::Mat &, cv::Mat &, cv::Mat &, int &);
double getSimilarity(const cv::Mat &, const cv::Mat &);
} // namespace opencv_utils

#endif