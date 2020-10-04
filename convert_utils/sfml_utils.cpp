/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** sfml_utils.hcpp
*/

#include "sfml_utils.hpp"

// Convert SFML image to Mat from Opencv
inline cv::Mat my::sfc::SFML2Mat(const sf::Image &img)
{
    cv::Size size(static_cast<int>(img.getSize().x), static_cast<int>(img.getSize().y));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
    cv::Mat mat(size, CV_8UC4, (void *)img.getPixelsPtr(), cv::Mat::AUTO_STEP);
#pragma GCC diagnostic pop
    cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);
    return mat.clone();
}

inline void my::sfc::SFML2Mat(const sf::Image &img, cv::Mat &mat)
{
    cv::Size size(static_cast<int>(img.getSize().x), static_cast<int>(img.getSize().y));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
    // mat.release();
    mat = cv::Mat(size, CV_8UC4, (void *)img.getPixelsPtr(), cv::Mat::AUTO_STEP);
#pragma GCC diagnostic pop
    cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);
}

// Convert Mat from Opencv to SFML image
inline sf::Image my::sfc::Mat2SFML(const cv::Mat &src)
{
    cv::Mat tmp;
    sf::Image image;
    cv::cvtColor(src, tmp, cv::COLOR_BGR2RGBA);
    image.create(static_cast<unsigned int>(tmp.cols), static_cast<unsigned int>(tmp.rows), tmp.ptr());
    tmp.release();
    return image;
}

inline void my::sfc::Mat2SFML(const cv::Mat &src, sf::Image &image)
{
    cv::Mat tmp;
    cv::cvtColor(src, tmp, cv::COLOR_BGR2RGBA);
    image.create(static_cast<unsigned int>(tmp.cols), static_cast<unsigned int>(tmp.rows), tmp.ptr());
    tmp.release();
}
