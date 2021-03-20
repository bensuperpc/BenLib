
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
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef MY_CUDA_MATRIX_HPP
#define MY_CUDA_MATRIX_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "stdio.h"

#ifndef BLOCK_SIZE
#    define BLOCK_SIZE 16
#endif

#define gpuErrchk(ans)                                                                                                                                         \
    {                                                                                                                                                          \
        gpuAssert((ans), __FILE__, __LINE__);                                                                                                                  \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

namespace my
{
namespace cuda
{

void matrixAddKernel(dim3 gridSize, dim3 blockSize, int *a, int *b, int *c, size_t n);
// void matrixAddKernel(dim3 gridSize, dim3 blockSize, cudaStream_t *streams, int *a, int *b, int *c, size_t n);

void matrixMultiplyShared(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, int n);
// void matrixMultiplyShared(dim3 gridSize, dim3 blockSize, cudaStream_t *streams, float *a, float *b, float *c, int n);
void matrixMultiplyShared(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols);

void matrixMut3D(dim3 gridSize, dim3 blockSize, int mat[][100][100]);

// 2D to 1D
template <typename T> void flatten1D(T **a, T *b, const size_t xMax, const size_t yMax);
// 3D to 1D
template <typename T> void flatten1D(T ***a, T *b, const size_t xMax, const size_t yMax, const size_t zMax);
// 4D to 1D
template <typename T> void flatten1D(T ****a, T *b, const size_t xMax, const size_t yMax, const size_t zMax, const size_t wMax);
// 1D to 2D
template <typename T> void reshape2D(T *a, T *b, const size_t xMax, const size_t yMax);
// 1D to 3D
template <typename T> void reshape3D(const T *a, T *b, const size_t xMax, const size_t yMax, const size_t zMax);
// 1D to 4D
template <typename T> void reshape4D(const T *a, T *b, const size_t xMax, const size_t yMax, const size_t zMax, const size_t wMax);

template <typename T> void cpu_matrix_mult(T *h_a, T *h_b, T *h_result, const size_t m);
template <typename T> void cpu_matrix_mult(T *matA, size_t rA, size_t cA, T *matB, size_t rB, size_t cB, T *matC, size_t rC, size_t cC);
} // namespace cuda
} // namespace my

#endif