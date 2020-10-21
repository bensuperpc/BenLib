/*
** BENSUPERPC PROJECT, 2020
** Filesystem
** File description:
** filesystem.cpp
*/

#include "filesystem.hpp"


void my::filesystem::filename_from_str(std::string & path)
{
    const size_t last_slash_idx = path.find_last_of("\\/");
    if (std::string::npos != last_slash_idx)
    {
        path.erase(0, last_slash_idx + 1);
    }
}

std::string my::filesystem::filename_from_str(const std::string & path)
{
    std::string filename = std::string(path);
    const size_t last_slash_idx = filename.find_last_of("\\/");
    if (std::string::npos != last_slash_idx)
    {
        filename.erase(0, last_slash_idx + 1);
    }
    return filename;
}

#if (__cplusplus == 201103L || __cplusplus == 201402L)

size_t my::filesystem::count_files(const std::string &path)
{
    size_t i = size_t(0);
    for (const auto &entry : fs::recursive_directory_iterator(path)) {
            i++;
    }
    return i;
}

void my::filesystem::list_all_files(std::vector<std::string> &list, const std::string &path)
{
    for (const auto &entry : fs::recursive_directory_iterator(path)) {
            list.emplace_back(entry.path().string());
    }
}

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

size_t my::filesystem::count_files(std::string_view path)
{
    size_t i = 0;
    for (const auto &entry : fs::recursive_directory_iterator(path)) {
            i++;
    }
    return i;
}

void my::filesystem::list_all_files(std::vector<std::string> &list, std::string_view path)
{
    for (const auto &entry : fs::recursive_directory_iterator(path)) {
            list.emplace_back(entry.path().string());
    }
}

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