/*
** BENSUPERPC PROJECT, 2020
** Texture
** File description:
** sfml_screenshot.hpp
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
