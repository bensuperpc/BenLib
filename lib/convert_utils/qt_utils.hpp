/*
** BENSUPERPC PROJECT, 2020
** RPG
** Source: -
** qt_utils.hpp
*/

#ifndef QT_UTILS_HPP
#define QT_UTILS_HPP

#include <QImage>
#include <boost/predef.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#if BOOST_COMP_GNUC
#endif

const std::map<int, int> Mat2QImageMap = {{CV_8UC4, QImage::Format_RGB32}, {CV_8UC3, QImage::Format_RGB888}, {CV_8UC4, QImage::Format_ARGB32}};
const std::map<int, int> QImage2MatMap = {{QImage::Format_RGB32, CV_8UC4}, {QImage::Format_RGB888, CV_8UC3}, {QImage::Format_ARGB32, CV_8UC4}};

namespace my
{
namespace qt_utils
{
QImage Mat2QImage(cv::Mat const &);
void Mat2QImage(cv::Mat const &, QImage &);
cv::Mat QImage2Mat(QImage const &src);
void QImage2Mat(QImage const &, cv::Mat &);
} // namespace qt_utils
} // namespace my
#endif
