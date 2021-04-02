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

/**
 * @brief ssfml namespace
 * @namespace ssfml
 */
namespace ssfml
{
/**
 * @brief 
 * 
 * @param window 
 * @param filename 
 * @return sf::Image 
 */
sf::Image take_screenshot(const sf::RenderWindow *window, const std::string &filename);

/**
 * @brief 
 * 
 * @param window 
 * @return sf::Image 
 */
sf::Image take_screenshot(const sf::RenderWindow *window);
} // namespace ssfml

#endif
