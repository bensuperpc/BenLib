/**
 * @file gettersetter.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2024-07-14
 *
 * MIT License
 *
 */

#ifndef BENLIB_PATERN_GETTERSETTER_HPP_
#define BENLIB_PATERN_GETTERSETTER_HPP_

#include <type_traits>
#include <utility>

#define GETTERSETTER(type, methodName, varName) \
    private: \
    type varName; \
    public: \
    const type& get##methodName() const { return varName; } \
    void set##methodName(const type& value) { varName = value; } \
    void set##methodName(type&& value) { varName = std::move(value); }

#endif  // BENLIB_PATERN_NONCOPYABLE_HPP_
