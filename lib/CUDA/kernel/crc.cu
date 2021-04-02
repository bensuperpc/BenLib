//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  BenLib, 2021                                            //
//  Created: 21, March, 2021                                //
//  Modified: 25, March, 2021                               //
//  file: kernel.cu                                         //
//  Crypto                                                  //
//  Source: https://stackoverflow.com/questions/13553015/cuda-c-linker-error-undefined-reference                                                //
//          https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/                                                //
//          https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation__concurrent-copy-and-execute
//          https://www.ce.jhu.edu/dalrymple/classes/602/Class12.pdf                                                //
//          https://create.stephan-brumme.com/crc32/
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

extern "C"
{
// Can be remove if use "external *Function*" in .c/cpp file
#include "crc.h"
}
#include "crc.cuh"
#include "crc.hpp"

__global__ void CRC32_byte_tableless_kernel(uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    uint crc = ~previousCrc32; // same as previousCrc32 ^ 0xFFFFFFFF

    while (length-- != 0) {
        uchar s = (uchar)crc ^ *data++;
        uint low = (s ^ (s << 6)) & 0xFF;
        uint a = (low * ((1 << 23) + (1 << 14) + (1 << 2)));
        crc = (crc >> 8) ^ (low * ((1 << 24) + (1 << 16) + (1 << 8))) ^ a ^ (a >> 1) ^ (low * ((1 << 20) + (1 << 12))) ^ (low << 19) ^ (low << 17) ^ (low >> 2);
    }

    *resultCrc32 = ~crc;
}

void my::cuda::CRC32_byte_tableless(const dim3 &grid, const dim3 &threads, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    CRC32_byte_tableless_kernel<<<grid, threads>>>(data, length, previousCrc32, resultCrc32);
}

void my::cuda::CRC32_byte_tableless(
    const dim3 &grid, const dim3 &threads, cudaStream_t &stream, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    CRC32_byte_tableless_kernel<<<grid, threads, 0, stream>>>(data, length, previousCrc32, resultCrc32);
}

extern "C" void CRC32_byte_tableless(const dim3 &grid, const dim3 &threads, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    CRC32_byte_tableless_kernel<<<grid, threads>>>(data, length, previousCrc32, resultCrc32);
}

__global__ void JAMCRC_byte_tableless_kernel(uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    uint crc = ~previousCrc32; // same as previousCrc32 ^ 0xFFFFFFFF

    while (length-- != 0) {
        uchar s = (uchar)crc ^ *data++;
        uint low = (s ^ (s << 6)) & 0xFF;
        uint a = (low * ((1 << 23) + (1 << 14) + (1 << 2)));
        crc = (crc >> 8) ^ (low * ((1 << 24) + (1 << 16) + (1 << 8))) ^ a ^ (a >> 1) ^ (low * ((1 << 20) + (1 << 12))) ^ (low << 19) ^ (low << 17) ^ (low >> 2);
    }

    *resultCrc32 = crc;
}

void my::cuda::JAMCRC_byte_tableless(const dim3 &grid, const dim3 &threads, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    JAMCRC_byte_tableless_kernel<<<grid, threads>>>(data, length, previousCrc32, resultCrc32);
}

void my::cuda::JAMCRC_byte_tableless(
    const dim3 &grid, const dim3 &threads, cudaStream_t &stream, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    JAMCRC_byte_tableless_kernel<<<grid, threads, 0, stream>>>(data, length, previousCrc32, resultCrc32);
}

extern "C" void JAMCRC_byte_tableless(const dim3 &grid, const dim3 &threads, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    JAMCRC_byte_tableless_kernel<<<grid, threads>>>(data, length, previousCrc32, resultCrc32);
}

__global__ void CRC32_byte_tableless2_kernel(uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    uint crc = ~previousCrc32;

    while (length-- != 0) {
        crc = crc ^ *data++;
        uint c = (((crc << 31) >> 31) & ((POLY >> 7) ^ (POLY >> 1))) ^ (((crc << 30) >> 31) & ((POLY >> 6) ^ POLY)) ^ (((crc << 29) >> 31) & (POLY >> 5))
                 ^ (((crc << 28) >> 31) & (POLY >> 4)) ^ (((crc << 27) >> 31) & (POLY >> 3)) ^ (((crc << 26) >> 31) & (POLY >> 2))
                 ^ (((crc << 25) >> 31) & (POLY >> 1)) ^ (((crc << 24) >> 31) & POLY);
        crc = (crc >> 8) ^ c;
    }
    *resultCrc32 = ~crc;
}

void my::cuda::CRC32_byte_tableless2(const dim3 &grid, const dim3 &threads, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    CRC32_byte_tableless2_kernel<<<grid, threads>>>(data, length, previousCrc32, resultCrc32);
}

void my::cuda::CRC32_byte_tableless2(
    const dim3 &grid, const dim3 &threads, cudaStream_t &stream, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    CRC32_byte_tableless2_kernel<<<grid, threads, 0, stream>>>(data, length, previousCrc32, resultCrc32);
}

extern "C" void CRC32_byte_tableless2(const dim3 &grid, const dim3 &threads, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    CRC32_byte_tableless2_kernel<<<grid, threads>>>(data, length, previousCrc32, resultCrc32);
}

__global__ void JAMCRC_byte_tableless2_kernel(uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    uint crc = ~previousCrc32;

    while (length-- != 0) {
        crc = crc ^ *data++;
        uint c = (((crc << 31) >> 31) & ((POLY >> 7) ^ (POLY >> 1))) ^ (((crc << 30) >> 31) & ((POLY >> 6) ^ POLY)) ^ (((crc << 29) >> 31) & (POLY >> 5))
                 ^ (((crc << 28) >> 31) & (POLY >> 4)) ^ (((crc << 27) >> 31) & (POLY >> 3)) ^ (((crc << 26) >> 31) & (POLY >> 2))
                 ^ (((crc << 25) >> 31) & (POLY >> 1)) ^ (((crc << 24) >> 31) & POLY);
        crc = (crc >> 8) ^ c;
    }
    *resultCrc32 = crc;
}

void my::cuda::JAMCRC_byte_tableless2(const dim3 &grid, const dim3 &threads, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    JAMCRC_byte_tableless2_kernel<<<grid, threads>>>(data, length, previousCrc32, resultCrc32);
}

void my::cuda::JAMCRC_byte_tableless2(
    const dim3 &grid, const dim3 &threads, cudaStream_t &stream, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    JAMCRC_byte_tableless2_kernel<<<grid, threads, 0, stream>>>(data, length, previousCrc32, resultCrc32);
}

extern "C" void JAMCRC_byte_tableless2(const dim3 &grid, const dim3 &threads, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32)
{
    JAMCRC_byte_tableless2_kernel<<<grid, threads>>>(data, length, previousCrc32, resultCrc32);
}