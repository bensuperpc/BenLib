/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** sfml_utils.hpp
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
cv::Mat SFML2Mat(const sf::Image &);
void SFML2Mat(const sf::Image &, cv::Mat &);

sf::Image Mat2SFML(const cv::Mat &);
void Mat2SFML(const cv::Mat &, sf::Image &);
} // namespace sfc
} // namespace my
#endif
