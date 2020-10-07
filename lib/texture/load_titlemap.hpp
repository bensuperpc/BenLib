/*
** BENSUPERPC PROJECT, 2020
** Texture
** File description:
** load_titlemap.hpp
*/

#ifndef LOAD_TITLEMAP_HPP_
#define LOAD_TITLEMAP_HPP_

#include <SFML/Graphics.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace title
{
void load_titlemap(std::vector<std::vector<size_t>> &, const std::string &);

} // namespace title

#endif
