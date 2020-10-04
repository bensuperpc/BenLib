/*
** BENSUPERPC PROJECT, 2020
** Math
** Source: https://stackoverflow.com/questions/178265/what-is-the-most-hard-to-understand-piece-of-c-code-you-know https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
** cpu.cpp
*/

#ifndef CPU_HPP_
#define CPU_HPP_

#include <iostream>
#include <string>

#ifdef _WIN32
#    include <intrin.h>
#else
#    include <x86intrin.h>
#endif

using namespace std;

namespace my
{
namespace cpu
{
#pragma GCC diagnostic ignored "-Wundef"
#if (__i386__ || _M_IX86 || __x86_64__ || _M_AMD64)

uint64_t rdtsc();

#endif
} // namespace cpu

} // namespace my
// https://stackoverflow.com/questions/2901694/how-to-detect-the-number-of-physical-processors-cores-on-windows-mac-and-linu
// https://stackoverflow.com/questions/21369381/measuring-cache-latencies
// https://stackoverflow.com/questions/13772567/how-to-get-the-cpu-cycle-count-in-x86-64-from-c
#endif
