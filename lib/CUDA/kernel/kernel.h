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
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef MY_CUDA_H
#define MY_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "stdio.h"

#ifndef BLOCK_SIZE
#    define BLOCK_SIZE 16
#endif

// 1D vector
void vecAdd(size_t gridSize, size_t blockSize, double *a, double *b, double *c, size_t n);
void vecSub(size_t gridSize, size_t blockSize, double *a, double *b, double *c, size_t n);
void vecMult(size_t gridSize, size_t blockSize, double *a, double *b, double *c, size_t n);
void vecDiv(size_t gridSize, size_t blockSize, double *a, double *b, double *c, size_t n);

// 2D vector
void matrixMultiplySimple(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, size_t n);
void matrixMultiplyOptimised(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, size_t n);
void multiply(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, int n);
#endif
