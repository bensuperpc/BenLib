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
//  file: vec1d.c                                           //
//  Crypto                                                  //
//  Source:                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

// main.cpp
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"

// extern void vecAdd(size_t gridSize, size_t blockSize, double *a, double *b, double *c, int n);

int main(int argc, char *argv[])
{
    // Size of vectors
    int n = 100000;

    // Host input vectors
    double *h_a;
    double *h_b;
    // Host output vector
    double *h_c;

    // Device input vectors
    double *d_a;
    double *d_b;
    // Device output vector
    double *d_c;

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(double);

    // Allocate memory for each vector on host
    h_a = (double *)malloc(bytes);
    h_b = (double *)malloc(bytes);
    h_c = (double *)malloc(bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc((void *)&d_a, bytes);
    cudaMalloc((void *)&d_b, bytes);
    cudaMalloc((void *)&d_c, bytes);

    int i;
    // Initialize vectors on host
    for (i = 0; i < n; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n / blockSize);

    // Execute the kernel
    // vecAdd_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    vecAdd(gridSize, blockSize, d_a, d_b, d_c, n);

    // Copy array back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for (i = 0; i < n; i++)
        sum += h_c[i];
    printf("final result: %f\n", sum / n);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
