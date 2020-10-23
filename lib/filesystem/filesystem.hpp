/*
** BENSUPERPC PROJECT, 2020
** Filesystem
** File description:
** filesystem.hpp
*/

// https://stackoverflow.com/questions/62409409/how-to-make-stdfilesystemdirectory-iterator-to-list-filenames-in-order
// https://stackoverflow.com/a/8520815/10152334
// https://stackoverflow.com/a/46931770/10152334

#ifndef FILE_SYSTEM_HPP_
#define FILE_SYSTEM_HPP_

#if defined(__GNUC__) || defined(__clang__)
#    define DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#    define DEPRECATED __declspec(deprecated)
#else
#    pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#    define DEPRECATED
#endif

#include <filesystem>
#include <vector>

#if (__cplusplus == 201103L || __cplusplus == 201402L)
#    define BOOST_FILESYSTEM_VERSION 3
#    define BOOST_FILESYSTEM_NO_DEPRECATED
#    include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#elif __cplusplus >= 201703L
namespace fs = std::filesystem;
#else
#    error This library needs at least a C++11 or above compliant compiler
#endif

namespace my
{
namespace filesystem
{

void filename_from_str(std::string &);

std::string filename_from_str(const std::string &);

#if (__cplusplus == 201103L || __cplusplus == 201402L)

size_t count_files(const std::string &);

void list_all_files(std::vector<std::string> &, const std::string &);

void search_by_name(std::vector<std::string> &, const std::string &, const std::string &);

void search_by_ext(std::vector<std::string> &, const std::string &, const std::string &);

void search_by_ext_and_name(std::vector<std::string> &, const std::string &, const std::string &, const std::string &);

#elif __cplusplus >= 201703L

size_t count_files(std::string_view);

void list_all_files(std::vector<std::string> &, std::string_view);

void search_by_name(std::vector<std::string> &, std::string_view, std::string_view);

void search_by_ext(std::vector<std::string> &, std::string_view, std::string_view);

void search_by_ext_and_name(std::vector<std::string> &, std::string_view, std::string_view, std::string_view);

#else

#    error This library needs at least a C++11 or above compliant compiler

#endif

} // namespace filesystem
} // namespace my
#endif
