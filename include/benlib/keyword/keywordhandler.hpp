/**
 * @file KeywordHandler.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */

#ifndef BENLIB_KEYWORD_KEYWORDHANDLER_HPP_
#define BENLIB_KEYWORD_KEYWORDHANDLER_HPP_

#include <any>
#include <functional>
#include <string>
#include <unordered_map>

class KeywordHandler
{

public:
    KeywordHandler() {}
    ~KeywordHandler() {}

    // Add signal with keyword and function
    template <typename... Args>
    void addKeyword(std::string keyword, void(*func)(Args...))
    {
        keywordmap.insert(std::make_pair(keyword, std::any(std::function<void(Args...)>(func))));
    }

    // Add signal with keyword and function
    template <typename... Args>
    void addKeyword(std::string keyword, std::function<void(Args...)> func)
    {
        keywordmap.insert(std::make_pair(keyword, std::any(func)));
    }

    // Call all signals with same keyword and arguments
    template <typename... Args>
    void callKeyword(std::string keyword, Args... args)
    {
        auto range = keywordmap.equal_range(keyword);
        for (auto it = range.first; it != range.second; ++it)
        {
            if (it->second.type() != typeid(std::function<void(Args...)>))
            {
                continue;   
            }

            std::any_cast<std::function<void(Args...)>>(it->second)(std::forward<Args>(args)...);
        }
    }

    // Remove all signals with same keyword
    std::size_t removeKeyword(std::string keyword)
    {
        return keywordmap.erase(keyword);
    }

    // Remove all signals with same keyword and same arguments
    template <typename... Args>
    void removeKeyword(std::string keyword, Args... args) {
        auto range = keywordmap.equal_range(keyword);
        for (auto it = range.first; it != range.second; ++it)
        {
            if (it->second.type() != typeid(std::function<void(Args...)>))
            {
                continue;   
            }

            if (std::any_cast<std::function<void(Args...)>>(it->second) == std::function<void(Args...)>(args...))
            {
                keywordmap.erase(it);
            }
        }
    }

    void removeAllSignal()
    {
        keywordmap.clear();
    }

    std::unordered_multimap<std::string, std::any>& data()
    {
        return keywordmap;
    }

private:
    std::unordered_multimap<std::string, std::any> keywordmap;

};

#endif  // BENLIB_KEYWORD_KEYWORDHANDLER_HPP_
