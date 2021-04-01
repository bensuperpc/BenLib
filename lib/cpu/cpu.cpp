/**
 * @file cpu.cpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#include "cpu.hpp"

#if (__i386__ || __x86_64__ || _M_AMD64)
/*
#    ifdef _WIN32
uint64_t my::cpu::rdtsc()
{
    return __rdtsc();
}
#    else
*/
uint64_t my::cpu::rdtsc()
{
    /*unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo; */
    return __rdtsc();
}

//#    endif

#endif