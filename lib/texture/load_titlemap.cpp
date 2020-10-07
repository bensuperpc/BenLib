/*
** BENSUPERPC PROJECT, 2020
** Texture
** File description:
** load_titlemap.cpp
*/

#include "load_titlemap.hpp"

void title::load_titlemap(std::vector<std::vector<size_t>> &title_map, const std::string &file)
{

    std::vector<size_t> _title_map;
    title_map.reserve(20);
    size_t number;

    std::string line;
    std::ifstream myfile(file);
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            std::stringstream iss(line);
            _title_map.reserve(20);
            while (iss >> number)
                _title_map.emplace_back(number);
            title_map.emplace_back(_title_map);
            _title_map.clear();
        }
        myfile.close();

        // Free unused memory
        _title_map.shrink_to_fit();
        title_map.shrink_to_fit();
    } else {
        std::cout << "Unable to open file";
    }
}