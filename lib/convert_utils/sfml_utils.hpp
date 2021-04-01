/**
 * @file sfml_utils.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#ifndef SFML_UTILS_HPP
#define SFML_UTILS_HPP

#include <SFML/Graphics.hpp>

// Disable Warning from OpenCV libs
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Winline"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#pragma GCC diagnostic pop

namespace my
{
namespace sfc
{
/**
 * @brief 
 * 
 * @param img 
 * @return cv::Mat 
 */
cv::Mat SFML2Mat(const sf::Image &img);
/**
 * @brief 
 * 
 * @param img 
 * @param mat 
 */
void SFML2Mat(const sf::Image &img, cv::Mat &mat);

/**
 * @brief 
 * 
 * @param src 
 * @return sf::Image 
 */
sf::Image Mat2SFML(const cv::Mat &src);

/**
 * @brief 
 * 
 * @param src 
 * @param img 
 */
void Mat2SFML(const cv::Mat &src, sf::Image &img);
} // namespace sfc
} // namespace my
#endif
