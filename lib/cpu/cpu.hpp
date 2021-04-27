/**
 * @file cpu.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

/*
** Source: https://stackoverflow.com/questions/178265/what-is-the-most-hard-to-understand-piece-of-c-code-you-know https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
*/

#ifndef CPU_HPP_
#define CPU_HPP_

#include <iostream>
#include <string>
extern "C"
{
#if defined(__x86_64__) || defined(__i386__)
#    ifdef _WIN32
#        include <intrin.h>
#    else
#        include <x86intrin.h>
#    endif
#endif
}

using namespace std;

namespace my
{
namespace cpu
{
#if defined(__i386__) || defined(__x86_64__) || defined(_M_AMD64)

/**
 * @brief Return nbrs cycle cpu
 * 
 * @return uint64_t 
 */
uint64_t rdtsc();

#endif
} // namespace cpu

} // namespace my
// https://stackoverflow.com/questions/2901694/how-to-detect-the-number-of-physical-processors-cores-on-windows-mac-and-linu
// https://stackoverflow.com/questions/21369381/measuring-cache-latencies
// https://stackoverflow.com/questions/13772567/how-to-get-the-cpu-cycle-count-in-x86-64-from-c
#endif
