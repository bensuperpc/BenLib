
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
//  file: kernel.cuh                                        //
//  Crypto                                                  //
//  Source: https://stackoverflow.com/questions/13553015/cuda-c-linker-error-undefined-reference                                                //
//          https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/                                                //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef MY_CUDA_MATRIX_CUH
#define MY_CUDA_MATRIX_CUH

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matrixAdd_kernel(int *matA, int *matB, int *matC, size_t N);

// 2D vector
__global__ void matrixMultiplyShared_kernel(float *left, float *right, float *res, int dim);
__global__ void matrixMultiplyShared_kernel(float *A, float *B, float *C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols);

__global__ void sharedABMultiply_kernel(float *a, float *b, float *c, int N);

__global__ void matFill_kernel(int *matA, int value, size_t sizeAX, size_t sizeAY);
__global__ void matCopy_kernel(int *matB, int *matA, int width, int height);

#endif