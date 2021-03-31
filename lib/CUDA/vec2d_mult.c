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
//  Created: 17, March, 2021                                //
//  Modified: 17, March, 2021                               //
//  file: OpenCL_test.cpp                                   //
//  Crypto                                                  //
//  Source: https://github.com/Kaixhin/cuda-workshop                                                 //
//          https://forums.developer.nvidia.com/t/double-pointer-allocation/9390                                                 //
//          https://stackoverflow.com/a/31382775/10152334                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"

const int TILE_WIDTH = 16;
#define THREADS_PER_BLOCK 1024

void matrixMultiplyCPU(float *a, float *b, float *c, size_t width);
void matrixMultiplyCPU(float *a, float *b, float *c, size_t width)
{
    float result;

    for (size_t row = 0; row < width; row++) {
        for (size_t col = 0; col < width; col++) {
            result = 0;
            for (size_t k = 0; k < width; k++) {
                result += a[row * width + k] * b[k * width + col];
            }
            c[row * width + col] = result;
        }
    }
}

void matrixMultiplyCPU_MP(float *a, float *b, float *c, size_t width);
void matrixMultiplyCPU_MP(float *a, float *b, float *c, size_t width)
{
    float result = 0.0;
//#pragma omp parallel
#pragma omp parallel for collapse(2) private(result)
    for (size_t row = 0; row < width; row++) {
        for (size_t col = 0; col < width; col++) {
            result = 0;
            for (size_t k = 0; k < width; k++) {
                result += a[row * width + k] * b[k * width + col];
            }
            c[row * width + col] = result;
        }
    }
}

int main()
{
    size_t width = 2000; // Define width of square matrix
    // Initialise grid and block variables
    int sqrtThreads = sqrt(THREADS_PER_BLOCK);
    size_t nBlocks = width / sqrtThreads;
    if (width % sqrtThreads != 0) { // Add an extra block if necessary
        nBlocks++;
    }
    int i_mult = 1;
    // dim3 grid(nBlocks, nBlocks, i_mult);
    dim3 grid = {nBlocks, nBlocks, i_mult};
    // dim3 block(sqrtThreads, sqrtThreads, i_mult); // Max number of threads per block
    dim3 block = {sqrtThreads, sqrtThreads, i_mult};
    // Initialise host pointers (dynamically allocated memory) and device pointers
    float *a_h;
    float *b_h;
    float *c_h; // GPU results
    float *d_h; // CPU results
    float *a_d;
    float *b_d;
    float *c_d;

    size_t size; // Number of bytes required by arrays

    // Create timer
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsed1, elapsed2, elapsed3;

    // Print out information about blocks and threads
    printf("Number of threads: %i (%ix%i)\n", block.x * block.y, block.x, block.y);
    printf("Number of blocks: %i (%ix%i)\n", grid.x * grid.y, grid.x, grid.y);

    // Dynamically allocate host memory
    size = width * width * sizeof(float);

    a_h = (float *)malloc(size);
    b_h = (float *)malloc(size);
    c_h = (float *)malloc(size);
    d_h = (float *)malloc(size);

    // Load host arrays with data
    for (size_t i = 0; i < width; i++) {
        for (size_t j = 0; j < width; j++) {
            a_h[i * width + j] = i;
            b_h[i * width + j] = i;
        }
    }

    // Allocate device memory
    cudaMalloc((void **)&a_d, size);
    cudaMalloc((void **)&b_d, size);
    cudaMalloc((void **)&c_d, size);

    // Copy host memory to device memory
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h, size, cudaMemcpyHostToDevice);

    // Start timer for GPU
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch kernel
    // matrixMultiplySimple<<<grid, block>>>(a_d, b_d, c_d, width);
    matrixMultiplySimple(grid, block, a_d, b_d, c_d, width);
    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed1, start, stop);

    // Print execution time
    printf("Time to calculate results on GPU: %f ms\n", elapsed1);

    // Copy results to host
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    // Start timer for CPU
    cudaEventRecord(start, 0);

    // Launch CPU code
    matrixMultiplyCPU_MP(a_h, b_h, d_h, width);

    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed2, start, stop);

    // Print execution time
    printf("Time to calculate results on CPU: %f ms\n", elapsed2);

    // Compare results
    for (size_t i = 0; i < width * width; i++) {
        if (c_h[i] != d_h[i]) {
            printf("Error: CPU and GPU results do not match\n");
            break;
        }
    }

    // Start timer for GPU (optimised)
    cudaEventRecord(start, 0);

    // Launch kernel (optimised)
    // matrixMultiplyOptimised<<<grid, block>>>(a_h, b_h, c_h, width);
    matrixMultiplyOptimised(grid, block, a_d, b_d, c_d, width);

    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed3, start, stop);

    // Print execution time
    printf("Time to calculate results on GPU (optimised): %f ms\n", elapsed3);

    // Copy results to host
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    // Compare results
    for (size_t i = 0; i < width * width; i++) {
        if (c_h[i] != d_h[i]) {
            printf("Error: CPU and GPU (optimised) results do not match\n");
            break;
        }
    }

    // Free memory
    free(a_h);
    free(b_h);
    free(c_h);
    free(d_h);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
