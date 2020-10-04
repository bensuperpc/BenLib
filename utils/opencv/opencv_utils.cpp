/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** opencv_utils.cpp
*/

#include "opencv_utils.hpp"

cv::Mat opencv_utils::cropimgrect(const cv::Mat &src, const cv::Rect &rect)
{
    cv::Mat croppedRef(src, rect);
    cv::Mat cropped;
    croppedRef.copyTo(cropped);
    return cropped;
}

void opencv_utils::cropimgrect(const cv::Mat &src, cv::Mat &des, const cv::Rect &rect)
{
    cv::Mat croppedRef(src, rect);
    cv::Mat cropped;
    croppedRef.copyTo(des);
}

void opencv_utils::imgdiff_prev(cv::Mat &img1, cv::Mat &img2, cv::Mat &out)
{
    int th = 10;  // 0
    imgdiff_prev(img1, img2, out, th);
}

void opencv_utils::imgdiff_prev(cv::Mat &img1, cv::Mat &img2, cv::Mat &out, int &th)
{
    if (img1.empty() || img2.empty())
        throw "imgdiff_prev: Empty image(s)";
    if (img1.rows != img2.rows || img1.cols != img2.cols)
        throw "imgdiff_prev: Image doesn't have same size";
    cv::Mat diff;
    absdiff(img1, img2, diff);

    cv::Mat mask(img1.size(), CV_8UC1);
    for(size_t j=0; j<diff.rows; ++j) {
        for(size_t i=0; i<diff.cols; ++i){
            cv::Vec3b pix = diff.at<cv::Vec3b>(j,i);
            long long int val = (pix[0] + pix[1] + pix[2]);
            if(val>th){
                mask.at<unsigned char>(j,i) = 255;
            }
        }
    }
    bitwise_and(img2, img2, out, mask);
}

double opencv_utils::getSimilarity(const cv::Mat &A, const cv::Mat &B)
{
    cv::Mat diff;
    cv::absdiff(A, B, diff);
    cv::threshold(diff, diff, 5, 255, cv::THRESH_BINARY);

    int64 pixelsdiff = 0;
    for (int j = 0; j < diff.rows; ++j) {
        for (int i = 0; i < diff.cols; ++i) {
            if (diff.ptr<cv::Vec3b>(j)[i][0] == 255) {
                pixelsdiff++;
            }
        }
    }
    return static_cast<double>(((diff.rows * diff.cols) / pixelsdiff) * 100.0);
}