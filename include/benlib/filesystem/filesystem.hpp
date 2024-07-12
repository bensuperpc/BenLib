/**
 * @file filesystem.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2024-07-07
 *
 * MIT License
 *
 */

#ifndef BENLIB_FILESYSTEM_HPP_
#define BENLIB_FILESYSTEM_HPP_

#include <vector>
#include <string_view>
#include <filesystem>

namespace benlib {
namespace filesystem {

static inline std::vector<std::string> listAllFiles(std::string_view path) {
    std::vector<std::string> list;
    for (const auto & entry : std::filesystem::directory_iterator(path))
    {
        list.push_back(entry.path());
    }

    return list;
}

static inline std::vector<std::string> listAllFiles(std::string_view path, std::string_view ext) {
    std::vector<std::string> list;
    for (const auto & entry : std::filesystem::directory_iterator(path))
    {
        if (entry.path().extension() == ext)
        {
            list.push_back(entry.path());
        }
    }

    return list;
}


}  // namespace filesystem
}  // namespace benlib
#endif

