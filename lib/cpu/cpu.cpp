/*
** BENSUPERPC PROJECT, 2020
** CPU
** File description:
** cpu.cpp
*/

#include "cpu.hpp"

#pragma GCC diagnostic ignored "-Wundef"
#if (__i386__ || _M_IX86 || __x86_64__ || _M_AMD64)
#    ifdef _WIN32
uint64_t rdtsc()
{
    return __rdtsc();
}
#    else

uint64_t my::cpu::rdtsc()
{
    /*unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo; */
    return __rdtsc();
}

#    endif

#endif