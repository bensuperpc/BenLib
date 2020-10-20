/*
** BENSUPERPC PROJECT, 2020
** Filesystem
** File description:
** filesystem.cpp
*/

#include "filesystem.hpp"

#if (__cplusplus == 201103L || __cplusplus == 201402L)

void my::filesystem::search_by_name(std::vector<std::string> &list, const std::string &path, const std::string &name)
{
    for (const auto &entry : fs::recursive_directory_iterator(path)) {
        if (entry.path().filename().string().find(name) != std::string::npos) {
            list.emplace_back(entry.path().string());
        }
    }
}

void my::filesystem::search_by_ext(std::vector<std::string> &list, const std::string &path, const std::string &ext)
{
    for (const auto &entry : fs::recursive_directory_iterator(path)) {
        if (entry.path().extension() == ext) {
            list.emplace_back(entry.path().string());
        }
    }
}

void my::filesystem::search_by_ext_and_name(std::vector<std::string> &list, const std::string &path, const std::string &ext, const std::string &name)
{
    for (const auto &entry : fs::recursive_directory_iterator(path)) {
        if (entry.path().extension() == ext) {
            if (entry.path().filename().string().find(name) != std::string::npos) {
                list.emplace_back(entry.path().string());
            }
        }
    }
}

#elif __cplusplus >= 201703L

void my::filesystem::search_by_name(std::vector<std::string> &list, std::string_view path, std::string_view name)
{
    for (const auto &entry : fs::recursive_directory_iterator(path)) {
        if (entry.path().filename().string().find(name) != std::string::npos) {
            list.emplace_back(entry.path().string());
        }
    }
}

void my::filesystem::search_by_ext(std::vector<std::string> &list, std::string_view path, std::string_view ext)
{
    for (const auto &entry : fs::recursive_directory_iterator(path)) {
        if (entry.path().extension() == ext) {
            list.emplace_back(entry.path().string());
        }
    }
}

void my::filesystem::search_by_ext_and_name(std::vector<std::string> &list, std::string_view path, std::string_view ext, std::string_view name)
{
    for (const auto &entry : fs::recursive_directory_iterator(path)) {
        if (entry.path().extension() == ext) {
            if (entry.path().filename().string().find(name) != std::string::npos) {
                list.emplace_back(entry.path().string());
            }
        }
    }
}

#else

#    error This library needs at least a C++11 or above compliant compiler

#endif