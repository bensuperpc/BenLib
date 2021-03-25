
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
//  Created: 20, March, 2021                                //
//  Modified: 20, March, 2021                               //
//  file: kernel.h                                          //
//  Crypto                                                  //
//  Source: https://stackoverflow.com/questions/13553015/cuda-c-linker-error-undefined-reference                                                //
//          https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/                                                //
//          https://gist.github.com/AndiH/2e2f6cd9bccd64ec73c3b1d2d18284e0
//          https://stackoverflow.com/a/14038590/10152334
//          https://www.daniweb.com/programming/software-development/threads/292133/convert-1d-array-to-2d-array
//          https://www.daniweb.com/programming/software-development/threads/471477/equivalent-iteration-of-2d-and-3d-array-flattened-as-1d-array
//          http://coliru.stacked-crooked.com/a/7c570672c13ca3bf
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef MY_CUDA_MATRIX_OPS_HPP
#define MY_CUDA_MATRIX_OPS_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
extern "C"
{
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

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

template <typename T> void copy(T ***B_, int ***A_, const size_t sizeX_, const size_t sizeY_, const size_t sizeZ_);
template <typename T> void copy(T **B_, int **A_, const size_t sizeX_, const size_t sizeY_);
template <typename T> void display(T ***A_, const size_t sizeX_, const size_t sizeY_, const size_t sizeZ_);
template <typename T> void display(T **A_, const size_t sizeX_, const size_t sizeY_);
template <typename T> void display(T *A_, const size_t sizeX_);
template <typename T> T ****aalloc(const size_t sizeX_, const size_t sizeY_, const size_t sizeZ_, const size_t sizeW_);
template <typename T> T ***aalloc(const size_t sizeX_, const size_t sizeY_, const size_t sizeZ_);
template <typename T> T **aalloc(const size_t sizeX_, const size_t sizeY_);
template <typename T> T *aalloc(const size_t sizeX_);
template <typename T> void adealloc(T ****A_, const size_t sizeX_, const size_t sizeY_, const size_t sizeZ_, const size_t sizeW_);
template <typename T> void adealloc(T ***A_, const size_t sizeX_, const size_t sizeY_, const size_t sizeZ_);
template <typename T> void adealloc(T **A_, const size_t sizeX_, const size_t sizeY_);
template <typename T> void adealloc(T *A_, const size_t sizeX_);
template <typename T> void adealloc(T *A_);

template <typename T> void print_matrices(T *matrix, char *file_Name, T x_dim, size_t y_dim, size_t dim);

template <typename T> int matRandFill(T **matA, dim3 &dimsA);

template <typename T>
int mMatAlloc(T **matA, T **matB, T **matC, const dim3 &dimsA, const dim3 &dimsB, dim3 &dimsC, const bool Unified_memory, const bool Pinned_memory,
    const bool set_memset);
template <typename T> int mMatAlloc(T **matA, T **matB, T **matC, const dim3 &dimsA, const dim3 &dimsB, dim3 &dimsC);

template <typename T>
int mMatAlloc(T **matC, const dim3 &dimsA, const dim3 &dimsB, const dim3 &dimsC, const bool Unified_memory, const bool Pinned_memory, const bool set_memset);
} // namespace cuda
} // namespace my

#endif