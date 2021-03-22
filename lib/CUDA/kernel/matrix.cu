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
//  file: kernel.cu                                         //
//  Crypto                                                  //
//  Source: https://stackoverflow.com/questions/13553015/cuda-c-linker-error-undefined-reference                                                //
//          https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/                                                //
//          https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation__concurrent-copy-and-execute
//          https://www.ce.jhu.edu/dalrymple/classes/602/Class12.pdf                                                //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

extern "C"
{
// Can be remove if use "external *Function*" in .c/cpp file
#include "matrix.h"
}
#include "matrix.cuh"
#include "matrix.hpp"

__global__ void matrixAddKernel_kernel(int *a, int *b, int *c, size_t N)
{
    size_t col = threadIdx.x + blockIdx.x * blockDim.x;
    size_t row = threadIdx.y + blockIdx.y * blockDim.y;
    size_t index = row * N + col;
    if (col < N && row < N) {
        c[index] = a[index] + b[index];
    }
    //__syncthreads();
}

void my::cuda::matrixAddKernel(dim3 gridSize, dim3 blockSize, int *a, int *b, int *c, size_t n)
{
    matrixAddKernel_kernel<<<gridSize, blockSize>>>(a, b, c, n);
    //cudaStreamSynchronize(0);
    // cudaDeviceSynchronize();
}

void my::cuda::matrixAddKernel(dim3 gridSize, dim3 blockSize, cudaStream_t stream, int *a, int *b, int *c, size_t n)
{
    matrixAddKernel_kernel<<<gridSize, blockSize, 0,stream>>>(a, b, c, n);
    //cudaStreamSynchronize(0);
    // cudaDeviceSynchronize();
}

extern "C" void matrixAddKernel(dim3 gridSize, dim3 blockSize, int *a, int *b, int *c, size_t n)
{
    matrixAddKernel_kernel<<<gridSize, blockSize>>>(a, b, c, n);
    //cudaStreamSynchronize(0);
    // cudaDeviceSynchronize();
}

__global__ void matrixMultiplyShared_kernel(float *left, float *right, float *res, int dim)
{

    unsigned int i, j;
    float temp = 0;

    __shared__ float Left_shared_t[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Right_shared_t[BLOCK_SIZE][BLOCK_SIZE];

    // Row i of matrix left
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {

        // Column j of matrix left
        j = tileNUM * BLOCK_SIZE + threadIdx.x;
        i = tileNUM * BLOCK_SIZE + threadIdx.y;
        // Load left[i][j] to shared mem

        Left_shared_t[threadIdx.y][threadIdx.x] = left[row * dim + j]; // Coalesced access
        // Load right[i][j] to shared mem

        Right_shared_t[threadIdx.y][threadIdx.x] = right[i * dim + col]; // Coalesced access
        // Synchronize before computation
        __syncthreads();

        // Accumulate one tile of res from tiles of left and right in shared mem
        for (unsigned int k = 0; k < BLOCK_SIZE; k++) {

            temp += Left_shared_t[threadIdx.y][k] * Right_shared_t[k][threadIdx.x]; // no shared memory bank conflict
        }
        // Synchronize
        __syncthreads();
    }
    // Store accumulated value to res
    res[row * dim + col] = temp;
}

void my::cuda::matrixMultiplyShared(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, int n)
{
    matrixMultiplyShared_kernel<<<gridSize, blockSize>>>(a, b, c, n);
    // cudaStreamSynchronize(0);
}

void my::cuda::matrixMultiplyShared(dim3 gridSize, dim3 blockSize, cudaStream_t stream, float *a, float *b, float *c, int n)
{
    matrixMultiplyShared_kernel<<<gridSize, blockSize, 0, stream>>>(a, b, c, n);
    // cudaStreamSynchronize(0);
}


extern "C" void matrixMultiplyShared(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, int n)
{
    matrixMultiplyShared_kernel<<<gridSize, blockSize>>>(a, b, c, n);
    // cudaStreamSynchronize(0);
}

/*
__global__ void matrixMultiplyShared_kernel(float *A, float *B, float *C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
{
    float CValue = 0;
    unsigned int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    unsigned int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    for (unsigned int k = 0; k < (BLOCK_SIZE + ACols - 1) / BLOCK_SIZE; k++) {
        if (k * BLOCK_SIZE + threadIdx.x < ACols && Row < ARows)
            As[threadIdx.y][threadIdx.x] = A[Row * ACols + k * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;
        if (k * BLOCK_SIZE + threadIdx.y < BRows && Col < BCols)
            Bs[threadIdx.y][threadIdx.x] = B[(k * BLOCK_SIZE + threadIdx.y) * BCols + Col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads();
        for (int n = 0; n < BLOCK_SIZE; ++n)
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
        __syncthreads();
    }
    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y) * CCols) + (blockIdx.x * blockDim.x) + threadIdx.x] = CValue;
}

void my::cuda::matrixMultiplyShared(
    dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
{
    matrixMultiplyShared_kernel<<<gridSize, blockSize>>>(a, b, c, ARows, ACols, BRows, BCols, CRows, CCols);
    cudaStreamSynchronize(0);
}
*/
/*
void my::cuda::matrixMultiplyShared(dim3 gridSize, dim3 blockSize, cudaStream_t *streams, float *a, float *b, float *c, int ARows, int ACols, int BRows, int
BCols, int CRows, int CCols)
{
    matrixMultiplyShared_kernel<<<gridSize, blockSize, 0, streams>>>(a, b, c, ARows, ACols, BRows, BCols, CRows, CCols);
}
*/

/*

__global__ void matrixMultiplyShared_kernel(float *left, float *right, float *res, int dim)
{

    unsigned int i, j, w;
    float temp = 0;

    __shared__ float Left_shared_t[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Right_shared_t[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];

    // Row i of matrix left
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int depth = threadIdx.z + blockIdx.z * blockDim.z;

    // Check image dimensions
    printf("Current x: %d\n", threadIdx.x);
    printf("Current y: %d\n", threadIdx.y);
    printf("Current z: %d\n", threadIdx.z);

    ////x + y * xMax + z * xMax * yMax
    for (unsigned int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {

        // Column j of matrix left
        j = tileNUM * BLOCK_SIZE + threadIdx.x;
        i = tileNUM * BLOCK_SIZE + threadIdx.y;
        w = tileNUM * BLOCK_SIZE + threadIdx.z;
        // Load left[i][j] to shared mem

        Left_shared_t[threadIdx.z][threadIdx.y][threadIdx.x] = left[row * dim + j + dim * dim * w]; // Coalesced access
        // Load right[i][j] to shared mem

        Right_shared_t[threadIdx.z][threadIdx.y][threadIdx.x] = right[i * dim + col + dim * dim * w]; // Coalesced access
        // Synchronize before computation
        __syncthreads();

        // Accumulate one tile of res from tiles of left and right in shared mem
        for (unsigned int k = 0; k < BLOCK_SIZE; k++) {
            for (unsigned int n = 0; n < BLOCK_SIZE; n++) {
                temp += Left_shared_t[n][threadIdx.y][k] * Right_shared_t[n][k][threadIdx.x]; // no shared memory bank conflict
            }
        }
        // Synchronize
        __syncthreads();
    }
    // Store accumulated value to res
    res[row * dim + col + dim * dim * depth] = temp;
    //x + y*width + z*height*width + w*height*width*depth.
}
*/

#define DATAXSIZE 100
#define DATAYSIZE 100
#define DATAZSIZE 20

__global__ void set(int a[][100][100])
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z * blockDim.z + threadIdx.z;
    /*
    printf("Current x: %d\n", threadIdx.x);
    printf("Current y: %d\n", threadIdx.y);
    printf("Current z: %d\n", threadIdx.z);
    */

    if ((idx < (DATAXSIZE)) && (idy < (DATAYSIZE)) && (idz < (DATAZSIZE))) {
        a[idz][idy][idx] = idz + idy + idx;
    }
}

void my::cuda::matrixMut3D(dim3 gridSize, dim3 blockSize, int mat[][DATAYSIZE][DATAXSIZE])
{
    // matrixMultiplyShared_kernel<<<gridSize, blockSize>>>(a, b, c, ARows, ACols, BRows, BCols, CRows, CCols);
    set<<<gridSize, blockSize>>>(mat);
    cudaStreamSynchronize(0);
}