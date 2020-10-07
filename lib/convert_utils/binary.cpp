/*
** BENSUPERPC PROJECT, 2020
** RPG
** Source: https://stackoverflow.com/questions/32410186/convert-bool-array-to-int32-unsigned-int-and-double
** binary.hcpp
*/

#include "binary.hpp"

unsigned long long my::binary::make_bitSet(bool *flags[], size_t size)
{
    std::bitset<8 * sizeof(ULLONG_MAX)> bitSet;
    if (isLittleEndian())
        for (size_t i = 0; i < size; ++i)
            bitSet.set(i, *flags[size - i - 1]);
    else
        for (size_t i = 0; i < size; ++i)
            bitSet.set(i, *flags[i]);

    return bitSet.to_ullong();
}

int my::binary::bitArrayToInt32_big(const bool *arr[], int count)
{
    int ret = 0;
    int tmp;
    for (int i = 0; i < count; i++) {
        tmp = *arr[i];
        ret |= tmp << (count - i - 1);
    }
    return ret;
}