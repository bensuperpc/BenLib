
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
//  Created: 16, March, 2021                                //
//  Modified: 17, March, 2021                               //
//  file: kernel.h                                          //
//  Crypto                                                  //
//  Source: https://stackoverflow.com/questions/13553015/cuda-c-linker-error-undefined-reference                                                //
//          https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/                                                //
//          https://gist.github.com/AndiH/2e2f6cd9bccd64ec73c3b1d2d18284e0
//          https://stackoverflow.com/a/14038590/10152334
//          https://www.daniweb.com/programming/software-development/threads/292133/convert-1d-array-to-2d-array
//          https://www.daniweb.com/programming/software-development/threads/471477/equivalent-iteration-of-2d-and-3d-array-flattened-as-1d-array
//          http://coliru.stacked-crooked.com/a/7c570672c13ca3bf
//          https://math.stackexchange.com/questions/63074/is-there-a-3-dimensional-matrix-by-matrix-product
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef MY_CUDA_CRC_HPP
#define MY_CUDA_CRC_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "stdio.h"

#ifndef BLOCK_SIZE
#    define BLOCK_SIZE 16
#endif

#ifndef gpuErrchk
#    define gpuErrchk(ans)                                                                                                                                     \
        {                                                                                                                                                      \
            gpuAssert((ans), __FILE__, __LINE__);                                                                                                              \
        }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
#endif

namespace my
{
namespace cuda
{
void CRC32_byte_tableless(const dim3 &grid, const dim3 &threads, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32);
void CRC32_byte_tableless(const dim3 &grid, const dim3 &threads, cudaStream_t &stream, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32);

void JAMCRC_byte_tableless(const dim3 &grid, const dim3 &threads, cudaStream_t &stream, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32);
void JAMCRC_byte_tableless(const dim3 &grid, const dim3 &threads, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32);

void CRC32_byte_tableless2(const dim3 &grid, const dim3 &threads, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32);
void CRC32_byte_tableless2(const dim3 &grid, const dim3 &threads, cudaStream_t &stream, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32);

void JAMCRC_byte_tableless2(const dim3 &grid, const dim3 &threads, cudaStream_t &stream, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32);
void JAMCRC_byte_tableless2(const dim3 &grid, const dim3 &threads, uchar *data, ulong length, uint previousCrc32, uint *resultCrc32);
} // namespace cuda
} // namespace my

#endif