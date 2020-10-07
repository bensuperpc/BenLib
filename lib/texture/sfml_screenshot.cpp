/*
** BENSUPERPC PROJECT, 2020
** Texture
** File description:
** sfml_screenshot.cpp
*/

#include "sfml_screenshot.hpp"

sf::Image ssfml::take_screenshot(const sf::RenderWindow *window, const std::string &filename)
{
    sf::Texture texture;
    texture.create(window->getSize().x, window->getSize().y);
    texture.update(*window);
    sf::Image &&img = texture.copyToImage();

    if (img.saveToFile(filename)) {
        std::cout << "screenshot saved to " << filename << std::endl;
    }
    return img;
}

sf::Image ssfml::take_screenshot(const sf::RenderWindow *window)
{
    sf::Texture texture;
    texture.create(window->getSize().x, window->getSize().y);
    texture.update(*window);
    sf::Image &&img = texture.copyToImage();
    return img;
}
