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
//  Created: 19, March, 2021                                //
//  Modified: 19, March, 2021                               //
//  file: OpenCL_test.cpp                                   //
//  Crypto                                                  //
//  Source: https://github.com/Kaixhin/cuda-workshop                                                 //
//          https://forums.developer.nvidia.com/t/double-pointer-allocation/9390                                                 //
//          https://stackoverflow.com/a/31382775/10152334                                                 //
//          https://github.com/kberkay/Cuda-Matrix-Multiplication/blob/master/matrix_Multiplication.cu                                                 //
//          https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#optimize
//          https://medium.com/gpgpu/multi-gpu-programming-6768eeb42e2c
//          https://stackoverflow.com/questions/12924155/sending-3d-array-to-cuda-kernel
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matrix.hpp"

// set a 3D volume
// To compile it with nvcc execute: nvcc -O2 -o set3d set3d.cu
// define the data set size (cubic volume)
#define DATAXSIZE 100
#define DATAYSIZE 100
#define DATAZSIZE 20
// define the chunk sizes that each threadblock will work on
#define BLKXSIZE 32
#define BLKYSIZE 4
#define BLKZSIZE 4

// for cuda error checking

#define cudaCheckErrors(msg)                                                                                                                                   \
    do {                                                                                                                                                       \
        cudaError_t __err = cudaGetLastError();                                                                                                                \
        if (__err != cudaSuccess) {                                                                                                                            \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__);                                            \
            fprintf(stderr, "*** FAILED - ABORTING\n");                                                                                                        \
            return 1;                                                                                                                                          \
        }                                                                                                                                                      \
    } while (0)
// device function to set the 3D volume

int main(int argc, char *argv[])
{
    typedef int nRarray[DATAYSIZE][DATAXSIZE];
    const dim3 blockSize(BLKXSIZE, BLKYSIZE, BLKZSIZE);
    const dim3 gridSize(((DATAXSIZE + BLKXSIZE - 1) / BLKXSIZE), ((DATAYSIZE + BLKYSIZE - 1) / BLKYSIZE), ((DATAZSIZE + BLKZSIZE - 1) / BLKZSIZE));
    // overall data set sizes
    const int nx = DATAXSIZE;
    const int ny = DATAYSIZE;
    const int nz = DATAZSIZE;
    // pointers for data set storage via malloc
    nRarray *c;   // storage for result stored on host
    nRarray *d_c; // storage for result computed on device
                  // allocate storage for data set
    if ((c = (nRarray *)malloc((nx * ny * nz) * sizeof(int))) == 0) {
        fprintf(stderr, "malloc1 Fail \n");
        return 1;
    }
    // allocate GPU device buffers
    cudaMalloc((void **)&d_c, (nx * ny * nz) * sizeof(int));
    cudaCheckErrors("Failed to allocate device buffer");
    // compute result
    // set<<<gridSize,blockSize>>>(d_c);
    my::cuda::matrixMut3D(gridSize, blockSize, d_c);
    cudaCheckErrors("Kernel launch failure");
    // copy output data back to host

    cudaMemcpy(c, d_c, ((nx * ny * nz) * sizeof(int)), cudaMemcpyDeviceToHost);
    cudaCheckErrors("CUDA memcpy failure");
    // and check for accuracy
    for (unsigned i = 0; i < nz; i++)
        for (unsigned j = 0; j < ny; j++)
            for (unsigned k = 0; k < nx; k++)
                if (c[i][j][k] != (i + j + k)) {
                    printf("Mismatch at x= %d, y= %d, z= %d  Host= %d, Device = %d\n", i, j, k, (i + j + k), c[i][j][k]);
                    return 1;
                }
    printf("Results check!\n");
    free(c);
    cudaFree(d_c);
    cudaCheckErrors("cudaFree fail");
    return 0;
}
