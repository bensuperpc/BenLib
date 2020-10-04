/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** screen_save.hpp
*/

#ifndef _SCREEN_SAVE_H_
#define _SCREEN_SAVE_H_

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

#include <SFML/Graphics.hpp>
#include <iostream>
#include <list>
#include "../../convert_utils/sfml_utils.hpp"

class Screen_save {
  public:
    Screen_save();
    ~Screen_save();

    sf::Image take_screenshot(const sf::RenderWindow *, const std::string &);
    sf::Image take_screenshot(const sf::RenderWindow *);
    sf::Image take_screenshot();

    void write_video(std::list<cv::Mat> &, cv::VideoWriter &);
    void write_video(std::list<sf::Texture> &, cv::VideoWriter &);

    void add_frame(const sf::RenderWindow *window);

    void save_screenshot(const std::string &);
    void save_video(const std::string &filename);

    size_t index = 0;
    size_t nbrs_save_frame = 1000;

    std::list<sf::Texture> textureList {};

  private:
  protected:
};

#endif
