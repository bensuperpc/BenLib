/*
** BENSUPERPC PROJECT, 2020
** Texture
** File description:
** load_texturemap.hpp
*/

#ifndef LOAD_TEXTUREMAP_HPP_
#define LOAD_TEXTUREMAP_HPP_

#include <SFML/Graphics.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace my
{
namespace texture
{
#if __cplusplus <= 201402L

void load_texturemap(std::map<int, std::string> &, const std::string &);
void load_texturemap(std::map<int, std::string *> &, const std::string &);

void load_texturemap(std::unordered_map<int, std::string> &, const std::string &);
void load_texturemap(std::unordered_map<int, std::string *> &, const std::string &);

void load_texturemap(std::vector<std::pair<const int, const std::string>> &, const std::string &);
void load_texturemap(std::vector<std::pair<const int, std::string *>> &, const std::string &);

#elif __cplusplus >= 201703L
template <typename T> void load_texturemap(std::unordered_map<int, T> &, const std::string &);

template <typename T> void load_texturemap(std::map<int, T> &, const std::string &);

template <typename T> void load_texturemap(std::vector<std::pair<const int, T>> &, const std::string &);
#else
#endif

} // namespace texture
} // namespace my
// THANK https://stackoverflow.com/questions/55553003/using-stdconditional-with-non-convertible-types-raw-vs-pointer
#endif
