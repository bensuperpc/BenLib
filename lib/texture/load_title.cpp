//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  BenLib, 2020                                            //
//  Created: 11, Novemver, 2020                             //
//  Modified: 11, Novemver, 2020                            //
//  file: load_titlemap.cpp                                 //
//  Load link titles                                        //
//  Source: -                                               //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include "load_title.hpp"

#if __cplusplus <= 201402L
void my::title::emplaceTitle(std::vector<sf::RectangleShape *> &_titleList, std::vector<std::vector<size_t>> &_title_map,
    std::unordered_map<std::string, std::unique_ptr<sf::Texture>> &_textureUMap, std::unordered_map<int, std::string> &_textureumap,
    const size_t &_texture_size)
#elif __cplusplus >= 201703L
void my::title::emplaceTitle(std::vector<std::unique_ptr<sf::RectangleShape>> &_titleList, std::vector<std::vector<size_t>> &_title_map,
    std::unordered_map<std::string, std::unique_ptr<sf::Texture>> &_textureUMap, std::unordered_map<int, std::string> &_textureumap,
    const size_t &_texture_size)
#else
#endif
{
    for (size_t x = 0; x < _title_map.size(); x++) {
        for (size_t y = 0; y < _title_map[x].size(); y++) {
            try {
#if __cplusplus <= 201402L
                sf::RectangleShape *title = new sf::RectangleShape();
#elif __cplusplus >= 201703L
                std::unique_ptr<sf::RectangleShape> title = std::make_unique<sf::RectangleShape>();
#else
#endif
                auto &&its = _textureumap.find(_title_map[x][y]);
                if (its != _textureumap.end()) {
                    auto &&it = _textureUMap.find(its->second);
                    if (it == _textureUMap.end()) {
                        std::cout << "Key-value pair not present in map" << std::endl;
                    } else {
#if __cplusplus <= 201402L
                        title->setTexture(it->second);
#elif __cplusplus >= 201703L
                        title->setTexture(it->second.get());
#else
#endif
                    }
                }
                title->setPosition(static_cast<float>(_texture_size * y), static_cast<float>(_texture_size * x));
                title->setSize(sf::Vector2f(static_cast<float>(_texture_size), static_cast<float>(_texture_size)));
                // title->setTextureRect(sf::IntRect(0, 0, (int)it->second.getSize().x, (int)it->second.getSize().y);
#if __cplusplus <= 201402L
                _titleList.emplace_back(title);
#elif __cplusplus >= 201703L
                _titleList.emplace_back(std::move(title));
#else
#endif
            }
            catch (...) {
                std::cout << "failed to load texture, wrong texture ID !" << std::endl;
            }
        }
    }
}

#if __cplusplus <= 201402L
void my::title::emplaceTitle(std::vector<sf::RectangleShape *> &_titleList, std::vector<std::vector<size_t>> &_title_map,
    std::map<std::string, std::unique_ptr<sf::Texture>> &_textureUMap, std::map<int, std::string> &_textureumap, const size_t &_texture_size)
#elif __cplusplus >= 201703L
void my::title::emplaceTitle(std::vector<std::unique_ptr<sf::RectangleShape>> &_titleList, std::vector<std::vector<size_t>> &_title_map,
    std::map<std::string, std::unique_ptr<sf::Texture>> &_textureUMap, std::map<int, std::string> &_textureumap, const size_t &_texture_size)
#else
#endif
{
    for (size_t x = 0; x < _title_map.size(); x++) {
        for (size_t y = 0; y < _title_map[x].size(); y++) {
            try {
#if __cplusplus <= 201402L
                sf::RectangleShape *title = new sf::RectangleShape();
#elif __cplusplus >= 201703L
                std::unique_ptr<sf::RectangleShape> title = std::make_unique<sf::RectangleShape>();
#else
#endif
                auto &&its = _textureumap.find(_title_map[x][y]);
                if (its != _textureumap.end()) {
                    auto &&it = _textureUMap.find(its->second);
                    if (it == _textureUMap.end()) {
                        std::cout << "Key-value pair not present in map" << std::endl;
                    } else {
#if __cplusplus <= 201402L
                        title->setTexture(it->second);
#elif __cplusplus >= 201703L
                        title->setTexture(it->second.get());
#else
#endif
                    }
                }
                title->setPosition(static_cast<float>(_texture_size * y), static_cast<float>(_texture_size * x));
                title->setSize(sf::Vector2f(static_cast<float>(_texture_size), static_cast<float>(_texture_size)));
                // title->setTextureRect(sf::IntRect(0, 0, (int)it->second.getSize().x, (int)it->second.getSize().y);
#if __cplusplus <= 201402L
                _titleList.emplace_back(title);
#elif __cplusplus >= 201703L
                _titleList.emplace_back(std::move(title));
#else
#endif
            }
            catch (...) {
                std::cout << "failed to load texture, wrong texture ID !" << std::endl;
            }
        }
    }
}

#if __cplusplus <= 201402L
void my::title::emplaceTitle(std::vector<sf::RectangleShape *> &_titleList, std::vector<std::vector<size_t>> &_title_map,
    std::vector<std::pair<std::string, std::unique_ptr<sf::Texture>>> &_textureList, std::vector<std::pair<int, std::string>> &_texturelist,
    const size_t &_texture_size)
#elif __cplusplus >= 201703L
void my::title::emplaceTitle(std::vector<std::unique_ptr<sf::RectangleShape>> &_titleList, std::vector<std::vector<size_t>> &_title_map,
    std::vector<std::pair<std::string, std::unique_ptr<sf::Texture>>> &_textureList, std::vector<std::pair<int, std::string>> &_texturelist,
    const size_t &_texture_size)
#else
#endif
{
    for (size_t x = 0; x < _title_map.size(); x++) {
        for (size_t y = 0; y < _title_map[x].size(); y++) {
            try {
#if __cplusplus <= 201402L
                sf::RectangleShape *title = new sf::RectangleShape();
#elif __cplusplus >= 201703L
                std::unique_ptr<sf::RectangleShape> title = std::make_unique<sf::RectangleShape>();
#else
#endif
                const int &&X = _title_map[x][y];
                auto &&it
                    = std::find_if(_texturelist.begin(), _texturelist.end(), [&X](const std::pair<int, std::string> &element) { return element.first == X; });
                if (it != _texturelist.end()) {
                    const std::string Y = it->second;
#if __cplusplus <= 201402L
                    auto &&its
                        = std::find_if(_textureList.begin(), _textureList.end(), [&Y](const std::pair<std::string, sf::Texture *> &t) { return t.first == Y; });
#elif __cplusplus >= 201703L
                    auto &&its = std::find_if(
                        _textureList.begin(), _textureList.end(), [&Y](const std::pair<std::string, std::unique_ptr<sf::Texture>> &t) { return t.first == Y; });
#else
#endif
                    if (its == _textureList.end()) {
                        std::cout << "Key-value pair not present in map" << std::endl;
                    } else {
#if __cplusplus <= 201402L
                        title->setTexture(its->second);
#elif __cplusplus >= 201703L
                        title->setTexture(its->second.get());
#else
#endif
                        title->setPosition(static_cast<float>(_texture_size * y), static_cast<float>(_texture_size * x));
                        title->setSize(sf::Vector2f(static_cast<float>(_texture_size), static_cast<float>(_texture_size)));
#if __cplusplus <= 201402L
                        _titleList.emplace_back(title);
#elif __cplusplus >= 201703L
                        _titleList.emplace_back(std::move(title));
#else
#endif
                    }
                } else {
                    std::cout << "Key-value pair not present in map:" << std::endl;
                }
            }
            catch (...) {
                std::cout << "failed to load texture, wrong texture ID !" << std::endl;
            }
        }
    }
}