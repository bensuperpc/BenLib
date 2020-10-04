/*
** BENSUPERPC PROJECT, 2020
** Math
** Source: https://stackoverflow.com/questions/178265/what-is-the-most-hard-to-understand-piece-of-c-code-you-know https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
** square_root.cpp
*/

#ifndef SQUARE_ROOT_HPP_
#define SQUARE_ROOT_HPP_

#ifdef __FAST_MATH__
#    error "-ffast-math is broken, don't use it"
#endif

#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>

#define MagicNBR_32 0x5f3759df
#define MagicNBR_64 0x5fe6eb50c7b537a9

namespace math
{

namespace square_root
{
float invsqrt(float x);
double invsqrt(double x);

float sqrt(float x);
double sqrt(double x);
template <typename T, char iterations = 2> inline T math::square_root::invsqrt(T);
} // namespace square_root
} // namespace math

// https://en.wikibooks.org/wiki/Optimizing_C%2B%2B/Code_optimization/Faster_operations
#endif
