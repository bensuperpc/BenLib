/*
** BENSUPERPC PROJECT, 2020
** Texture
** File description:
** load_titlemap.cpp
*/

#include "load_texturemap.hpp"

#if __cplusplus <= 201402L
void my::texture::load_texturemap(std::map<int, std::string> &texture_map, const std::string &file)
{
    std::string index = "";
    std::string path = "";
    std::string line;
    std::ifstream myfile(file);
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            std::stringstream iss(line);
            std::getline(iss, index, ',');
            iss >> path;
            texture_map.insert(std::pair<int, std::string>(std::stoi(index), path));
        }
    }
    myfile.close();
}

void my::texture::load_texturemap(std::map<int, std::string *> &texture_map, const std::string &file)
{
    std::string index = "";
    std::string path = "";
    std::string line;
    std::ifstream myfile(file);
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            std::stringstream iss(line);
            std::getline(iss, index, ',');
            iss >> path;
            std::string *pathptr = new std::string(path);
            texture_map.insert(std::pair<int, std::string *>(std::stoi(index), pathptr));
        }
    }
    myfile.close();
}

void my::texture::load_texturemap(std::unordered_map<int, std::string> &texture_map, const std::string &file)
{
    std::string index = "";
    std::string path = "";
    std::string line;
    std::ifstream myfile(file);
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            std::stringstream iss(line);
            std::getline(iss, index, ',');
            iss >> path;
            texture_map.insert(std::pair<int, std::string>(std::stoi(index), path));
        }
    }
    myfile.close();
}

void my::texture::load_texturemap(std::unordered_map<int, std::string *> &texture_map, const std::string &file)
{
    std::string index = "";
    std::string path = "";
    std::string line;
    std::ifstream myfile(file);
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            std::stringstream iss(line);
            std::getline(iss, index, ',');
            iss >> path;
            std::string *pathptr = new std::string(path);
            texture_map.insert(std::pair<int, std::string *>(std::stoi(index), pathptr));
        }
    }
    myfile.close();
}

void my::texture::load_texturemap(std::vector<std::pair<const int, const std::string>> &texture_map, const std::string &file)
{
    std::string index = "";
    std::string path = "";
    std::string line;
    std::ifstream myfile(file);
    if (myfile.is_open()) {
        texture_map.reserve(250);
        while (getline(myfile, line)) {
            std::stringstream iss(line);
            std::getline(iss, index, ',');
            iss >> path;
            texture_map.emplace_back(std::make_pair(std::stoi(index), path));
        }
        texture_map.shrink_to_fit();
    }
    myfile.close();
}

void my::texture::load_texturemap(std::vector<std::pair<const int, std::string *>> &texture_map, const std::string &file)
{
    std::string index = "";
    std::string path = "";
    std::string line;
    std::ifstream myfile(file);
    if (myfile.is_open()) {
        texture_map.reserve(250);
        while (getline(myfile, line)) {
            std::stringstream iss(line);
            std::getline(iss, index, ',');
            iss >> path;
            std::string *pathptr = new std::string(path);
            texture_map.emplace_back(std::make_pair(std::stoi(index), pathptr));
        }
        texture_map.shrink_to_fit();
    }
    myfile.close();
}

#elif __cplusplus >= 201703L

template <typename T> void my::texture::load_texturemap(std::unordered_map<int, T> &texture_map, const std::string &file)
{
    // typedef typename std::conditional<std::is_pointer<T>::value == true, std::string *, std::string>::type Ts;
    std::string index = "";
    std::string path = "";
    std::string line;
    std::ifstream myfile(file);
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            std::stringstream iss(line);
            std::getline(iss, index, ',');
            iss >> path;
            if constexpr (std::is_pointer<T>::value) {
                std::string *pathptr = new std::string(path);
                texture_map.insert(std::make_pair(std::stoi(index), pathptr));
            } else {
                texture_map.insert(std::make_pair(std::stoi(index), path));
            }
        }
    }
    myfile.close();
}
template void my::texture::load_texturemap<std::string>(std::unordered_map<int, std::string> &texture_map, const std::string &file);
template void my::texture::load_texturemap<std::string *>(std::unordered_map<int, std::string *> &texture_map, const std::string &file);

template <typename T> void my::texture::load_texturemap(std::map<int, T> &texture_map, const std::string &file)
{
    // typedef typename std::conditional<std::is_pointer<T>::value == true, std::string *, std::string>::type Ts;
    std::string index = "";
    std::string path = "";
    std::string line;
    std::ifstream myfile(file);
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            std::stringstream iss(line);
            std::getline(iss, index, ',');
            iss >> path;
            if constexpr (std::is_pointer<T>::value) {
                std::string *pathptr = new std::string(path);
                texture_map.insert(std::make_pair(std::stoi(index), pathptr));
            } else {
                texture_map.insert(std::make_pair(std::stoi(index), path));
            }
        }
    }
    myfile.close();
}
template void my::texture::load_texturemap<std::string>(std::map<int, std::string> &texture_map, const std::string &file);
template void my::texture::load_texturemap<std::string *>(std::map<int, std::string *> &texture_map, const std::string &file);

template <typename T> void my::texture::load_texturemap(std::vector<std::pair<const int, T>> &texture_map, const std::string &file)
{
    // typedef typename std::conditional<std::is_pointer<T>::value == true, std::string *, std::string>::type Ts;
    std::string index = "";
    std::string path = "";
    std::string line;
    std::ifstream myfile(file);
    if (myfile.is_open()) {
        texture_map.reserve(250);
        while (getline(myfile, line)) {
            std::stringstream iss(line);
            std::getline(iss, index, ',');
            iss >> path;
            if constexpr (std::is_pointer<T>::value) {
                std::string *pathptr = new std::string(path);
                texture_map.emplace_back(std::make_pair(std::stoi(index), pathptr));
            } else {
                texture_map.emplace_back(std::make_pair(std::stoi(index), path));
            }
        }
        texture_map.shrink_to_fit();
    }
    myfile.close();
}
template void my::texture::load_texturemap<std::string>(std::vector<std::pair<const int, std::string>> &texture_map, const std::string &file);
template void my::texture::load_texturemap<std::string *>(std::vector<std::pair<const int, std::string *>> &texture_map, const std::string &file);
#else
#endif