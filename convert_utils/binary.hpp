/*
** BENSUPERPC PROJECT, 2020
** RPG
** Source: https://stackoverflow.com/questions/32410186/convert-bool-array-to-int32-unsigned-int-and-double
** binary.hpp
*/

#ifndef BINARY_HPP
#define BINARY_HPP

#include <bitset>
#include <boost/predef.h>
#include <climits>
#include <iostream>
#include <vector>

enum class endianness : bool
{
    big,
    little
};

#if (BOOST_COMP_GNUC && __cplusplus >= 201402L)
constexpr endianness getEndianness()
{
    uint32_t word = 1;
    uint8_t *byte = (uint8_t *)&word;

    if (byte[0])
        return endianness::little;
    else
        return endianness::big;
}

constexpr bool isLittleEndian()
{
    switch (getEndianness()) {
        case endianness::little:
            return true;
        case endianness::big:
            return false;
        default:
            return false;
    }
}
#else
endianness getEndianness();
endianness getEndianness()
{
    uint32_t word = 1;
    uint8_t *byte = (uint8_t *)&word;

    if (byte[0])
        return endianness::little;
    else
        return endianness::big;
}
bool isLittleEndian();
bool isLittleEndian()
{
    switch (getEndianness()) {
        case endianness::little:
            return true;
        case endianness::big:
            return false;
        default:
            return false;
    }
}
#endif
namespace my
{
namespace binary
{
unsigned long long make_bitSet(bool *[], size_t);
int bitArrayToInt32_big(const bool *[], int);
template <typename T> T bitArrayToInt_big(const bool *ar[], size_t ar_size);
template <typename T> T bitArrayToInt_big(const std::vector<bool> &, size_t);

} // namespace binary
} // namespace my
#include "binary_imp.hpp"

#endif
