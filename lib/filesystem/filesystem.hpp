/**
 * @file filesystem.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
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
/**
 * @brief 
 * 
 * @param path 
 */
void filename_from_str(std::string & path);

/**
 * @brief 
 * 
 * @param path 
 * @return std::string 
 */
std::string filename_from_str(const std::string & path);

#if (__cplusplus == 201103L || __cplusplus == 201402L)
/**
 * @brief 
 * 
 * @param path 
 * @return size_t 
 */
size_t count_files(const std::string & path);

/**
 * @brief 
 * 
 * @param list 
 * @param path 
 */
void list_all_files(std::vector<std::string> &list, const std::string & path);

/**
 * @brief 
 * 
 * @param list 
 * @param path 
 * @param name 
 */
void search_by_name(std::vector<std::string> &list, const std::string &path, const std::string &name);

/**
 * @brief 
 * 
 * @param list 
 * @param path 
 * @param name 
 */
void search_by_ext(std::vector<std::string> &list, const std::string &path, const std::string &name);

/**
 * @brief 
 * 
 * @param list 
 */
void search_by_ext_and_name(std::vector<std::string> &list, const std::string &path, const std::string &name, const std::string &ext);

#elif __cplusplus >= 201703L
/**
 * @brief 
 * 
 * @return size_t 
 */
size_t count_files(std::string_view);

/**
 * @brief 
 * 
 * @param list 
 * @param path 
 */
void list_all_files(std::vector<std::string> &list, std::string_view path);

/**
 * @brief 
 * 
 * @param list 
 * @param path 
 * @param name 
 */
void search_by_name(std::vector<std::string> &list, std::string_view path, std::string_view name);

/**
 * @brief 
 * 
 * @param list 
 * @param path 
 * @param name 
 */
void search_by_ext(std::vector<std::string> &list, std::string_view path, std::string_view name);

/**
 * @brief 
 * 
 * @param list 
 * @param path 
 * @param name 
 * @param ext 
 */
void search_by_ext_and_name(std::vector<std::string> &list, std::string_view path, std::string_view name, std::string_view ext);

#else

#    error This library needs at least a C++11 or above compliant compiler

#endif

} // namespace filesystem
} // namespace my
#endif
