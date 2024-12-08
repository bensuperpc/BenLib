/**
 * @file random.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2021-04-01
 *
 * MIT License
 *
 */

/*
** Source:
*https://stackoverflow.com/questions/14638739/generating-a-random-double-between-a-range-of-values
*/

#ifndef BENLIB_MATH_RANDOM_HPP_
#define BENLIB_MATH_RANDOM_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
#if __has_include("omp.h")
#include <omp.h>
#endif
#endif

#if !defined(_OPENMP)
#if _MSC_VER && !__INTEL_COMPILER
#pragma message("No openMP ! Only use 1 thread.")
#else
#warning No openMP ! Only use 1 thread.
#endif
#endif

namespace benlib {
namespace math {
namespace rand {
/**
 * @brief
 *
 * @tparam T
 * @param lower
 * @param upper
 * @return T
 */
template <typename T, bool mersenne_64 = true>
auto random(const T& fMin, const T& fMax) -> T {
    typedef typename std::conditional<mersenne_64 == true, std::mt19937_64, std::mt19937>::type random_engine;
    random_engine rng;
    std::random_device rnd_device;
    rng.seed(rnd_device());

    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(fMin, fMax);
        return dist(rng);
    } else /*if (std::is_floating_point<T>::value)*/ {
        std::uniform_real_distribution<T> dist(fMin, fMax);
        return dist(rng);
    }
}

/**
 * @brief
 *
 * @tparam T
 * @param lower
 * @param upper
 */
template <typename T, bool mersenne_64 = true>
auto random(std::vector<T>& vec, const T& lower, const T& upper) -> void {
#if defined(_OPENMP)
#ifdef _MSC_VER
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
#endif
    for (typename std::vector<T>::size_type i = 0; i != vec.size(); i++) {
        vec[i] = random<T, mersenne_64>(lower, upper);
    }
}

/**
 * @brief
 *
 * @tparam T
 * @param lower
 * @param upper
 */
template <typename T, bool mersenne_64 = true>
auto random(T* arr, const std::size_t& S, const T& lower, const T& upper) -> void {
#if defined(_OPENMP)
#ifdef _MSC_VER
#pragma omp parallel for schedule(static)
#else
#pragma omp parallel for schedule(auto)
#endif
#endif
    for (std::size_t i = 0; i != S; i++) {
        arr[i] = random<T, mersenne_64>(lower, upper);
    }
}

}  // namespace rand
}  // namespace math
}  // namespace benlib
#endif
