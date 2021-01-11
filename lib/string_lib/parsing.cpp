/*
** BENSUPERPC PROJECT, 2020
** Filesystem
** File description:
** filesystem.cpp
*/

#include "parsing.hpp"

std::vector<std::string> my::string::split(const std::string &s, const std::string &delimiter)
{
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.emplace_back(token);
    }
    res.emplace_back(s.substr(pos_start));
    return res;
}

void my::string::split(std::vector<std::string> &res, const std::string &s, const std::string &delimiter)
{
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.emplace_back(token);
    }
    res.emplace_back(s.substr(pos_start));
}

std::vector<std::string> my::string::split(const std::string &s, const char delim)
{
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.emplace_back(item);
    }
    return result;
}

void my::string::split(std::vector<std::string> &result, const std::string &s, const char delim)
{
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.emplace_back(item);
    }
}

void my::string::csv_parse(std::vector<std::vector<std::string>> & file, const std::string &filename, const char delimiter)
{
    std::ifstream input_from_file(filename);
    std::string lines;
    std::vector<std::string> line;
    while (getline(input_from_file, lines)) {

        //lines += std::string(1, delimiter);

        std::stringstream ss(lines);

        std::string word;

        while (getline(ss, word, delimiter))
            line.emplace_back(word);
        file.emplace_back(line);
        line.clear();
    }
}