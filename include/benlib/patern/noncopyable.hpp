/**
 * @file noncopyable.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2024-07-14
 *
 * MIT License
 *
 */

#ifndef BENLIB_PATERN_NONCOPYABLE_HPP_
#define BENLIB_PATERN_NONCOPYABLE_HPP_

class NonCopyable {
   protected:
    NonCopyable() = default;
    NonCopyable(NonCopyable&&) = default;
    NonCopyable& operator=(NonCopyable&&) = default;
    virtual ~NonCopyable() = default;

    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;
};

#endif  // BENLIB_PATERN_NONCOPYABLE_HPP_
