/*
** BENSUPERPC PROJECT, 2020
** Math
** File description:
** count_digits.hpp
*/

#include "count_digits.hpp"

template <typename T> T my::math::count_digits::count_digits_1(T n)
{
    T count = 0;
    while (n != 0) {
        n = n / 10;
        ++count;
    }
    return count;
}

template <typename T> T my::math::count_digits::count_digits_2(T n)
{
    if (n == 0)
        return 0;
    return 1 + count_digits_2(n / 10);
}

template <typename T> T my::math::count_digits::count_digits_3(T n)
{
    return (T)std::floor(std::log10(n) + 1);
}

template <typename T> T my::math::count_digits::count_digits_4(T n)
{
    std::string num = std::to_string(n);
    return (T)num.size();
}