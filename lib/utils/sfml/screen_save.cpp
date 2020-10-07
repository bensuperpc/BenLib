/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** screen_save.cpp
*/

#include "screen_save.hpp"

Screen_save::Screen_save()
{ /*this->textureList.resize(this->nbrs_save_frame);*/
}

Screen_save::~Screen_save()
{
    std::cout << "OK1" << std::endl;
    // this->textureList.clear();
    std::cout << "OK2" << std::endl;
    // this->textureList.shrink_to_fit();
    std::cout << "OK3" << std::endl;
}

void Screen_save::add_frame(const sf::RenderWindow *window)
{
    textureList.emplace_back(sf::Texture());
    textureList.back().create(window->getSize().x, window->getSize().y);
    textureList.back().update(*window);
    if (textureList.size() >= nbrs_save_frame)
        textureList.pop_front();
}

void Screen_save::save_screenshot(const std::string &filename)
{

    cv::Mat im;
    sfc::SFML2Mat(this->take_screenshot(), im);
    cv::imwrite(filename, im);
    im.release();
}

void Screen_save::save_video(const std::string &filename)
{
    cv::VideoWriter video("out.avi", cv::VideoWriter::fourcc('X', '2', '6', '4'), 10,
        cv::Size(static_cast<int>(textureList.back().getSize().x), static_cast<int>(textureList.back().getSize().y)), true);
    this->write_video(this->textureList, video);
    video.release();
}

sf::Image Screen_save::take_screenshot()
{
    sf::Image &&img = textureList.back().copyToImage();
    return img;
}

sf::Image Screen_save::take_screenshot(const sf::RenderWindow *window, const std::string &filename)
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

sf::Image Screen_save::take_screenshot(const sf::RenderWindow *window)
{
    sf::Texture texture;
    texture.create(window->getSize().x, window->getSize().y);
    texture.update(*window);
    sf::Image &&img = texture.copyToImage();
    return img;
}

void Screen_save::write_video(std::list<cv::Mat> &vid, cv::VideoWriter &vidwriter)
{
    for (const auto &trame_vid : vid) {
        vidwriter.write(trame_vid);
    }
}

void Screen_save::write_video(std::list<sf::Texture> &vid, cv::VideoWriter &vidwriter)
{
    for (auto &trame_vid : vid) {
        vidwriter.write(sfc::SFML2Mat(trame_vid.copyToImage()));
    }
}