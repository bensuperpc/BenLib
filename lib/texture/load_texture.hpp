/*
** BENSUPERPC PROJECT, 2020
** Texture
** File description:
** load_texture.hpp
*/

#ifndef LOAD_TEXTURE_HPP_
#define LOAD_TEXTURE_HPP_

#if defined(__GNUC__) || defined(__clang__)
#    define DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#    define DEPRECATED __declspec(deprecated)
#else
#    pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#    define DEPRECATED
#endif

#include <SFML/Graphics.hpp>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>

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

#define RESERVE_VECTOR 500

namespace my
{
namespace texture
{
// Thank https://stackoverflow.com/a/21265197/10152334

#if (__cplusplus == 201103L || __cplusplus == 201402L)
// vector
/*DEPRECATED*/ void load_texture(std::vector<std::pair<const std::string, sf::Texture>> &, const std::string &);
/*DEPRECATED*/ void load_texture(std::vector<std::pair<const std::string, sf::Texture *>> &, const std::string &);
// map
/*DEPRECATED*/ void load_texture(std::map<const std::string, sf::Texture> &, const std::string &);
/*DEPRECATED*/ void load_texture(std::map<const std::string, sf::Texture *> &, const std::string &);

// unordered_map
void load_texture(std::unordered_map<std::string, sf::Texture> &, const std::string &);
void load_texture(std::unordered_map<std::string, sf::Texture *> &, const std::string &);
void load_texture(std::unordered_map<std::string, sf::Texture *> &, const std::string &, const bool);

#elif __cplusplus >= 201703L
// vector
void load_texture(std::vector<std::pair<const std::string, sf::Texture>> &, std::string_view);
void load_texture(std::vector<std::pair<const std::string, std::unique_ptr<sf::Texture>>> &, std::string_view);
void load_texture(std::vector<std::pair<const std::string, sf::Texture *>> &, std::string_view, const bool &);
void load_texture(std::vector<std::pair<const std::string, sf::Texture *>> &, std::string_view);

// map
void load_texture(std::map<const std::string, sf::Texture> &, std::string_view);
void load_texture(std::map<const std::string, sf::Texture *> &, std::string_view);
void load_texture(std::map<const std::string, std::unique_ptr<sf::Texture>> &, std::string_view);

// unordered_map
void load_texture(std::unordered_map<std::string, sf::Texture> &, std::string_view);
void load_texture(std::unordered_map<std::string, sf::Texture *> &, std::string_view);
void load_texture(std::unordered_map<std::string, sf::Texture *> &, std::string_view, const bool &);
void load_texture(std::unordered_map<std::string, std::unique_ptr<sf::Texture>> &, std::string_view);

#else

#    error This library needs at least a C++11 or above compliant compiler

#endif

} // namespace texture
} // namespace my
#endif
