/**
 * @file sfml_screenshot.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#ifndef SFML_SCREENSHOT_HPP_
#define SFML_SCREENSHOT_HPP_

#include <SFML/Graphics.hpp>
#include <iostream>

namespace ssfml
{
sf::Image take_screenshot(const sf::RenderWindow *, const std::string &);
sf::Image take_screenshot(const sf::RenderWindow *);
} // namespace ssfml

#endif
