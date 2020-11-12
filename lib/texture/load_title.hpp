//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  RPG, 2020                                               //
//  Created: 11, Novemver, 2020                             //
//  Modified: 11, Novemver, 2020                            //
//  file: load_titlemap.cpp                                 //
//  Load link titles                                        //
//  Source: -                                               //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef LOAD_TITLE_HPP_
#define LOAD_TITLE_HPP_

#include <SFML/Graphics.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
namespace my
{
namespace title
{

#if __cplusplus <= 201402L
void emplaceTitle(std::vector<sf::RectangleShape *> &, std::vector<std::vector<size_t>> &, std::unordered_map<std::string, std::unique_ptr<sf::Texture>> &,
    std::unordered_map<int, std::string> &, const size_t &);
void emplaceTitle(std::vector<sf::RectangleShape *> &, std::vector<std::vector<size_t>> &, std::map<std::string, std::unique_ptr<sf::Texture>> &,
    std::map<int, std::string> &, const size_t &);
void emplaceTitle(std::vector<sf::RectangleShape *> &, std::vector<std::vector<size_t>> &, std::vector<std::pair<std::string, std::unique_ptr<sf::Texture>>> &,
    std::vector<std::pair<int, std::string>> &, const size_t &);
#elif __cplusplus >= 201703L
void emplaceTitle(std::vector<std::unique_ptr<sf::RectangleShape>> &, std::vector<std::vector<size_t>> &,
    std::unordered_map<std::string, std::unique_ptr<sf::Texture>> &, std::unordered_map<int, std::string> &, const size_t &);
void emplaceTitle(std::vector<std::unique_ptr<sf::RectangleShape>> &, std::vector<std::vector<size_t>> &, std::map<std::string, std::unique_ptr<sf::Texture>> &,
    std::map<int, std::string> &, const size_t &);
void emplaceTitle(std::vector<std::unique_ptr<sf::RectangleShape>> &, std::vector<std::vector<size_t>> &,
    std::vector<std::pair<std::string, std::unique_ptr<sf::Texture>>> &, std::vector<std::pair<int, std::string>> &, const size_t &);
#elif __cplusplus >= 201703L

#else
#endif

} // namespace title
} // namespace my
#endif
