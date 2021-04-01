/**
 * @file binary_imp.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief https://stackoverflow.com/questions/32410186/convert-bool-array-to-int32-unsigned-int-and-double
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#include "binary.hpp"

template <typename T> T my::binary::bitArrayToInt_big(const bool *ar[], size_t ar_size)
{
    T ret {};

    for (size_t i = 0; i < ar_size; ++i) {
        T s {*ar[i]};
        s <<= i;
        ret |= s;
    }

    return ret;
}

template <typename T> T my::binary::bitArrayToInt_big(const std::vector<bool> &vec, size_t ar_size)
{
    T ret {};

    for (size_t i = 0; i < ar_size; ++i) {
        T s {vec[i]};
        s <<= i;
        ret |= s;
    }

    return ret;
}