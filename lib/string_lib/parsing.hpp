/**
 * @file parsing.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

// https://stackoverflow.com/a/46931770/10152334

#ifndef PARSING_HPP_
#define PARSING_HPP_

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace my
{
/**
 * @namespace my::string
 * @brief the string namespace, To do string ops
 */
namespace string
{
/**
 * @brief 
 * 
 * @param s 
 * @param delimiter 
 * @return std::vector<std::string> 
 */
std::vector<std::string> split(const std::string &s, const std::string &delimiter);

/**
 * @brief 
 * 
 * @param res 
 * @param s 
 * @param delimiter 
 */
void split(std::vector<std::string> &res, const std::string &s, const std::string &delimiter);

/**
 * @brief 
 * 
 * @param s 
 * @param delim 
 * @return std::vector<std::string> 
 */
std::vector<std::string> split(const std::string &s, const char delim);

/**
 * @brief 
 * 
 * @param s 
 * @param delim 
 * @return std::vector<std::string> 
 */
void split(std::vector<std::string> &, const std::string &s, const char);

/**
 * @brief 
 * 
 * @param file 
 * @param filename 
 * @param delimiter 
 */
void csv_parse(std::vector<std::vector<std::string>> &file, const std::string &filename, const char delimiter);
} // namespace string
} // namespace my
#endif
