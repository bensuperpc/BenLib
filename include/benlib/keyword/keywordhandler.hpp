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

class KeywordHandler {

public:
    KeywordHandler() {}
    virtual ~KeywordHandler() = default;

    // Add signal with keyword and function
    template <typename... Args>
    void addKeyword(std::string keyword, void(*func)(Args...)) {
        _keywordmap.insert(std::make_pair(keyword, std::any(std::function<void(Args...)>(func))));
    }

    // Add signal with keyword and function
    template <typename... Args>
    void addKeyword(std::string keyword, std::function<void(Args...)> func) {
        _keywordmap.insert(std::make_pair(keyword, std::any(func)));
    }

    // Call all signals with same keyword and arguments
    template <typename... Args>
    void callKeyword(std::string keyword, Args... args) {
        auto range = _keywordmap.equal_range(keyword);
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
    std::size_t removeKeyword(std::string keyword) {
        return _keywordmap.erase(keyword);
    }

    // Remove all signals with same keyword and same arguments
    template <typename... Args>
    void removeKeyword(std::string keyword, Args... args) {
        auto range = _keywordmap.equal_range(keyword);
        for (auto it = range.first; it != range.second; ++it)
        {
            if (it->second.type() != typeid(std::function<void(Args...)>))
            {
                continue;   
            }

            if (std::any_cast<std::function<void(Args...)>>(it->second) == std::function<void(Args...)>(args...))
            {
                _keywordmap.erase(it);
            }
        }
    }

    void removeAllSignal() {
        _keywordmap.clear();
    }

    std::unordered_multimap<std::string, std::any>& data() {
        return _keywordmap;
    }

protected:
    std::unordered_multimap<std::string, std::any> _keywordmap;
};

#endif  // BENLIB_KEYWORD_KEYWORDHANDLER_HPP_
