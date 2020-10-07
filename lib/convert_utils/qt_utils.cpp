/*
** BENSUPERPC PROJECT, 2020
** RPG
** Source: https://stackoverflow.com/questions/32410186/convert-bool-array-to-int32-unsigned-int-and-double
** qt_utils.cpp
*/

#include "qt_utils.hpp"

inline QImage my::qt_utils::Mat2QImage(cv::Mat const &src)
{
    cv::Mat temp;
    cvtColor(src, temp, cv::COLOR_RGB2BGR);
    QImage dest((const uchar *)temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
    dest.bits();
    return dest;
}

inline void my::qt_utils::Mat2QImage(cv::Mat const &src, QImage &dst)
{
    cv::Mat temp;
    cvtColor(src, temp, cv::COLOR_RGB2BGR);
    QImage *dest = new QImage((const uchar *)temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
    dest->bits();
    dst = dest->copy();
    delete dest;
}

inline cv::Mat my::qt_utils::QImage2Mat(QImage const &src)
{
    // utilisez "cv::Mat &&" pour eviter les copies inutiles
    cv::Mat tmp(src.height(), src.width(), CV_8UC3, (uchar *)src.bits(), static_cast<uint32_t>(src.bytesPerLine()));
    cv::Mat result;
    cvtColor(tmp, result, cv::COLOR_RGB2BGR);
    return result;
}

inline void my::qt_utils::QImage2Mat(QImage const &src, cv::Mat &dst)
{
    cv::Mat tmp(src.height(), src.width(), CV_8UC3, (uchar *)src.bits(), static_cast<uint32_t>(src.bytesPerLine()));
    cvtColor(tmp, dst, cv::COLOR_RGB2BGR);
}