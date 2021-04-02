/**
 * @file load_titlemap.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
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
/**
 * @brief 
 * 
 * @param title_map 
 * @param file 
 */
void load_titlemap(std::vector<std::vector<size_t>> &title_map, const std::string &file);

} // namespace title

#endif
