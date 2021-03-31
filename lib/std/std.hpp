/**
 * @file std.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

// https://stackoverflow.com/questions/17902405/how-to-implement-make-unique-function-in-c11
/**
 * @brief 
 * 
 */
#ifndef STD_HPP_
#define STD_HPP_

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

/**
 * @brief std namespace
 * 
 */
namespace std
{
/**
 * @brief 
 * 
 * @tparam T 
 */
template <class T> struct _Unique_if
{
    typedef unique_ptr<T> _Single_object;
};

/**
 * @brief 
 * 
 * @tparam T 
 */
template <class T> struct _Unique_if<T[]>
{
    typedef unique_ptr<T[]> _Unknown_bound;
};

/**
 * @brief 
 * 
 * @tparam T 
 * @tparam N 
 */
template <class T, size_t N> struct _Unique_if<T[N]>
{
    typedef void _Known_bound;
};

/**
 * @brief 
 * 
 * @tparam T 
 * @tparam Args 
 * @param args 
 * @return _Unique_if<T>::_Single_object 
 */
template <class T, class... Args> typename _Unique_if<T>::_Single_object make_unique(Args &&...args)
{
    return unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/**
 * @brief 
 * 
 * @tparam T 
 * @param n 
 * @return _Unique_if<T>::_Unknown_bound 
 */
template <class T> typename _Unique_if<T>::_Unknown_bound make_unique(size_t n)
{
    typedef typename remove_extent<T>::type U;
    return unique_ptr<T>(new U[n]());
}

/**
 * @brief 
 * 
 * @tparam T 
 * @tparam Args 
 * @return _Unique_if<T>::_Known_bound 
 */
template <class T, class... Args> typename _Unique_if<T>::_Known_bound make_unique(Args &&...) = delete;
} // namespace std
#endif
