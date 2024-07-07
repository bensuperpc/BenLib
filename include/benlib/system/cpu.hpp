/**
 * @file cpu.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2024-07-06
 *
 * MIT License
 *
 */

#ifndef BENLIB_SYSTEM_CPU_HPP_
#define BENLIB_SYSTEM_CPU_HPP_

#include <cstdint>

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

namespace benlib {
namespace system {
namespace cpu {

    uint64_t rdtsc() noexcept
    {
        #if defined(__x86_64__) || defined(__i386__)
        // x86_64 and i386 architecture
        unsigned int lo, hi;
        __asm__ __volatile__("rdtsc"
                             : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;

        #elif defined(__aarch64__)
        // ARM64 architecture
        uint64_t val;
        asm volatile("mrs %0, cntvct_el0" : "=r"(val));
        return val;

        #elif defined(__arm__)
        // ARM32 architecture
        uint32_t val;
        asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(val));
        return val;
        #endif
    }

}  // namespace cpu
}  // namespace system
}  // namespace benlib
#endif
