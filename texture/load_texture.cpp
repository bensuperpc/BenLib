/*
** BENSUPERPC PROJECT, 2020
** Texture
** File description:
** load_texture.cpp
*/

#include "load_texture.hpp"

#if (__cplusplus == 201103L || __cplusplus == 201402L)
void my::texture::load_texture(std::vector<std::pair<const std::string, sf::Texture>> &_textureList, const std::string &_path)
{
    if (fs::exists(_path) && fs::is_directory(_path)) {
        _textureList.reserve(RESERVE_VECTOR);
        for (auto const &entry : fs::recursive_directory_iterator(_path)) {
            if (fs::is_regular_file(entry) && entry.path().extension() == ".png") {
                sf::Texture texture;
                if (texture.loadFromFile(entry.path().string())) {
                    _textureList.emplace_back(std::make_pair(entry.path().string(), texture));
#    ifdef DNDEBUG
                    std::cout << entry.path().string() << std::endl;
#    endif
                } else {
                    std::cout << "Texture not found !" << std::endl;
                }
            }
        }
        _textureList.shrink_to_fit();
    }
}

void my::texture::load_texture(std::vector<std::pair<const std::string, sf::Texture *>> &_textureList, const std::string &_path)
{
    if (fs::exists(_path) && fs::is_directory(_path)) {
        _textureList.reserve(RESERVE_VECTOR);
        for (auto const &entry : fs::recursive_directory_iterator(_path)) {
            if (fs::is_regular_file(entry) && entry.path().extension() == ".png") {
                sf::Texture *texture = new sf::Texture();
                if (texture->loadFromFile(entry.path().string())) {
                    _textureList.emplace_back(std::make_pair(entry.path().string(), texture));
#    ifdef DNDEBUG
                    std::cout << entry.path().string() << std::endl;
#    endif
                } else {
                    std::cout << "Texture not found !" << std::endl;
                }
            }
        }
        _textureList.shrink_to_fit();
    }
}

void my::texture::load_texture(std::map<const std::string, sf::Texture> &_textureList, const std::string &_path)
{
    if (fs::exists(_path) && fs::is_directory(_path)) {
        for (auto const &entry : fs::recursive_directory_iterator(_path)) {
            if (fs::is_regular_file(entry) && entry.path().extension() == ".png") {
                sf::Texture texture;
                if (texture.loadFromFile(entry.path().string())) {
                    _textureList.insert(std::pair<const std::string, sf::Texture>(entry.path().string(), texture));
#    ifdef DNDEBUG
                    std::cout << entry.path().string() << std::endl;
#    endif
                } else {
                    std::cout << "Texture not found !" << std::endl;
                }
            }
        }
    }
}

void my::texture::load_texture(std::map<const std::string, sf::Texture *> &_textureList, const std::string &_path)
{
    if (fs::exists(_path) && fs::is_directory(_path)) {
        for (auto const &entry : fs::recursive_directory_iterator(_path)) {
            if (fs::is_regular_file(entry) && entry.path().extension() == ".png") {
                sf::Texture *texture = new sf::Texture();
                if (texture->loadFromFile(entry.path().string())) {
                    _textureList.insert(std::pair<const std::string, sf::Texture *>(entry.path().string(), texture));
#    ifdef DNDEBUG
                    std::cout << entry.path().string() << std::endl;
#    endif
                } else {
                    std::cout << "Texture not found !" << std::endl;
                }
            }
        }
    }
}

void my::texture::load_texture(std::unordered_map<std::string, sf::Texture> &_textureList, const std::string &_path)
{
    if (fs::exists(_path) && fs::is_directory(_path)) {
        for (auto const &entry : fs::recursive_directory_iterator(_path)) {
            if (fs::is_regular_file(entry) && entry.path().extension() == ".png") {
                sf::Texture texture;
                if (texture.loadFromFile(entry.path().string())) {
                    _textureList.insert(std::pair<std::string, sf::Texture>(entry.path().string(), texture));
#    ifdef DNDEBUG
                    std::cout << entry.path().string() << std::endl;
#    endif
                } else {
                    std::cout << "Texture not found !" << std::endl;
                }
            }
        }
    }
}

void my::texture::load_texture(std::unordered_map<std::string, sf::Texture *> &_textureList, const std::string &_path)
{
    load_texture(_textureList, _path, true);
}

void my::texture::load_texture(std::unordered_map<std::string, sf::Texture *> &_textureList, const std::string &_path, const bool load_texture)
{
    if (fs::exists(_path) && fs::is_directory(_path)) {
        for (const auto &entry : fs::recursive_directory_iterator(_path)) {
            if (entry.path().extension() == ".png") {
                sf::Texture *texture = new sf::Texture();
                if (load_texture == true) {
                    if (texture->loadFromFile(entry.path().string())) {
                        _textureList.insert(std::make_pair(entry.path().string(), texture));
                    } else {
                        std::cout << "Texture not found !" << std::endl;
                    }
                } else {
                    _textureList.insert(std::make_pair(entry.path().string(), texture));
                }
            }
#    ifdef DNDEBUG
            std::cout << entry.path().string() << std::endl;
#    endif
            //
        }
    }
}

#elif __cplusplus >= 201703L

void my::texture::load_texture(std::vector<std::pair<const std::string, sf::Texture>> &_textureList, std::string_view _path)
{
    _textureList.reserve(RESERVE_VECTOR);
    for (const auto &entry : fs::recursive_directory_iterator(_path)) {
        if (entry.path().extension() == ".png") {
            sf::Texture texture;
            if (texture.loadFromFile(entry.path())) {
                _textureList.emplace_back(std::make_pair(entry.path().string(), texture));
            } else {
                std::cout << "Texture not found !" << std::endl;
            }
#    ifdef DNDEBUG
            std::cout << entry.path() << std::endl;
#    endif
            //
        }
    }
    _textureList.shrink_to_fit();
}

void my::texture::load_texture(std::vector<std::pair<const std::string, std::unique_ptr<sf::Texture>>> &_textureList, std::string_view _path)
{
    _textureList.reserve(RESERVE_VECTOR);
    for (const auto &entry : fs::recursive_directory_iterator(_path)) {
        if (entry.path().extension() == ".png") {
            auto texture = std::make_unique<sf::Texture>();
            if (texture->loadFromFile(entry.path())) {
                _textureList.emplace_back(std::make_pair(entry.path().string(), std::move(texture)));
            } else {
                std::cout << "Texture not found !" << std::endl;
            }
            texture.reset();
#    ifdef DNDEBUG
            std::cout << entry.path() << std::endl;
#    endif
        }
    }
    _textureList.shrink_to_fit();
}

void my::texture::load_texture(std::vector<std::pair<const std::string, sf::Texture *>> &_textureList, std::string_view _path)
{
    load_texture(_textureList, _path, true);
}

void my::texture::load_texture(std::vector<std::pair<const std::string, sf::Texture *>> &_textureList, std::string_view _path, const bool &load_texture)
{
    _textureList.reserve(RESERVE_VECTOR);
    for (const auto &entry : fs::recursive_directory_iterator(_path)) {
        if (entry.path().extension() == ".png") {
            sf::Texture *texture = new sf::Texture();

            if (load_texture == true) {
                if (texture->loadFromFile(entry.path())) {
                    _textureList.emplace_back(std::make_pair(entry.path().string(), texture));
                } else {
                    std::cout << "Texture not found !" << std::endl;
                }
            } else {
                _textureList.emplace_back(std::make_pair(entry.path().string(), texture));
            }
#    ifdef DNDEBUG
            std::cout << entry.path() << std::endl;
#    endif
            //
        }
    }
    _textureList.shrink_to_fit();
}

void my::texture::load_texture(std::map<const std::string, sf::Texture> &_textureList, std::string_view _path)
{
    for (const auto &entry : fs::recursive_directory_iterator(_path)) {
        if (entry.path().extension() == ".png") {
            sf::Texture texture = sf::Texture();
            if (texture.loadFromFile(entry.path())) {
                _textureList.insert(std::pair<const std::string, sf::Texture>(entry.path().string(), texture));
            } else {
                std::cout << "Texture not found !" << std::endl;
            }
#    ifdef DNDEBUG
            std::cout << entry.path() << std::endl;
#    endif
            //
        }
    }
}

void my::texture::load_texture(std::map<const std::string, sf::Texture *> &_textureList, std::string_view _path)
{
    for (const auto &entry : fs::recursive_directory_iterator(_path)) {
        if (entry.path().extension() == ".png") {
            sf::Texture *texture = new sf::Texture();
            if (texture->loadFromFile(entry.path())) {
                _textureList.insert(std::pair<const std::string, sf::Texture *>(entry.path().string(), texture));
            } else {
                std::cout << "Texture not found !" << std::endl;
            }
#    ifdef DNDEBUG
            std::cout << entry.path().string() << std::endl;
#    endif
            //
        }
    }
}

void my::texture::load_texture(std::map<const std::string, std::unique_ptr<sf::Texture>> &_textureList, std::string_view _path)
{
    for (const auto &entry : fs::recursive_directory_iterator(_path)) {
        if (entry.path().extension() == ".png") {
            auto texture = std::make_unique<sf::Texture>();
            if (texture->loadFromFile(entry.path())) {
                _textureList.insert(std::pair<const std::string, std::unique_ptr<sf::Texture>>(entry.path().string(), std::move(texture)));
            } else {
                std::cout << "Texture not found !" << std::endl;
            }
            texture.reset();
#    ifndef DNDEBUG
            std::cout << entry.path() << std::endl;
#    endif
        }
    }
}

void my::texture::load_texture(std::unordered_map<std::string, sf::Texture> &_textureList, std::string_view _path)
{
    for (const auto &entry : fs::recursive_directory_iterator(_path)) {
        if (entry.path().extension() == ".png") {
            sf::Texture texture = sf::Texture();
            if (texture.loadFromFile(entry.path())) {
                _textureList.insert(std::make_pair(entry.path().string(), texture));
            } else {
                std::cout << "Texture not found !" << std::endl;
            }
        }
#    ifdef DNDEBUG
        std::cout << entry.path().string() << std::endl;
#    endif
        //
    }
}

void my::texture::load_texture(std::unordered_map<std::string, sf::Texture *> &_textureList, std::string_view _path)
{
    load_texture(_textureList, _path, true);
}

void my::texture::load_texture(std::unordered_map<std::string, sf::Texture *> &_textureList, std::string_view _path, const bool &load_texture)
{
    for (const auto &entry : fs::recursive_directory_iterator(_path)) {
        if (entry.path().extension() == ".png") {
            sf::Texture *texture = new sf::Texture();
            if (load_texture == true) {
                if (texture->loadFromFile(entry.path())) {
                    _textureList.insert(std::make_pair(entry.path().string(), texture));
                } else {
                    std::cout << "Texture not found !" << std::endl;
                }
            } else {
                _textureList.insert(std::make_pair(entry.path().string(), texture));
            }
        }
#    ifdef DNDEBUG
        std::cout << entry.path().string() << std::endl;
#    endif
        //
    }
}

void my::texture::load_texture(std::unordered_map<std::string, std::unique_ptr<sf::Texture>> &_textureList, std::string_view _path)
{
    for (const auto &entry : fs::recursive_directory_iterator(_path)) {
        if (entry.path().extension() == ".png") {
            sf::Texture *texture = new sf::Texture();
            if (texture->loadFromFile(entry.path())) {
                _textureList.insert(std::make_pair(entry.path().string(), std::move(texture)));
            } else {
                std::cout << "Texture not found !" << std::endl;
            }
        }
#    ifdef DNDEBUG
        std::cout << entry.path().string() << std::endl;
#    endif
        //
    }
}

#else

#    error This library needs at least a C++11 or above compliant compiler

#endif