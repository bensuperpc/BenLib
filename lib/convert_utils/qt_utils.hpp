/**
 * @file qt_utils.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief https://stackoverflow.com/questions/32410186/convert-bool-array-to-int32-unsigned-int-and-double
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#ifndef QT_UTILS_HPP
#define QT_UTILS_HPP

#include <QImage>
extern "C"
{
#include <boost/predef.h>
}
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

const std::map<int, int> Mat2QImageMap = {{CV_8UC4, QImage::Format_RGB32}, {CV_8UC3, QImage::Format_RGB888}, {CV_8UC4, QImage::Format_ARGB32}};
const std::map<int, int> QImage2MatMap = {{QImage::Format_RGB32, CV_8UC4}, {QImage::Format_RGB888, CV_8UC3}, {QImage::Format_ARGB32, CV_8UC4}};

namespace my
{
namespace qt_utils
{
inline QImage Mat2QImage(cv::Mat const &);
inline void Mat2QImage(cv::Mat const &, QImage &);
inline cv::Mat QImage2Mat(QImage const &src);
inline void QImage2Mat(QImage const &, cv::Mat &);
} // namespace qt_utils
} // namespace my
#endif
