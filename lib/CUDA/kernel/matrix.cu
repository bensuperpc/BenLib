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
//          https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
//          http://oz.nthu.edu.tw/~d947207/NVIDIA/copy3D/Matrix_transpose_post.pdf
//          https://on-demand.gputechconf.com/gtc/2017/presentation/s7122-stephen-jones-cuda-optimization-tips-tricks-and-techniques.pdf
//          https://github.com/NVIDIA/cuda-samples/blob/v11.2/Samples/matrixMul/matrixMul.cu
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

/**
 * \brief To mult 2D flat Matrix
 * \param int* matA 2D flat matrix
 * \param int* matB 2D flat matrix
 * \param int* matC 2D flat matrix
 * \param size_t size of matrix
 */
__global__ void matrixAdd_kernel(int *matA, int *matB, int *matC, size_t N)
{
    size_t col = threadIdx.x + blockIdx.x * blockDim.x;
    size_t row = threadIdx.y + blockIdx.y * blockDim.y;
    size_t index = row * N + col;
    if (col < N && row < N) {
        matC[index] = matA[index] + matB[index];
    }
    //__syncthreads();
}

void my::cuda::matrixAdd(dim3 gridSize, dim3 blockSize, int *a, int *b, int *c, size_t n)
{
    matrixAdd_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

void my::cuda::matrixAdd(dim3 gridSize, dim3 blockSize, cudaStream_t stream, int *a, int *b, int *c, size_t n)
{
    matrixAdd_kernel<<<gridSize, blockSize, 0, stream>>>(a, b, c, n);
}

extern "C" void matrixAdd(dim3 gridSize, dim3 blockSize, int *a, int *b, int *c, size_t n)
{
    matrixAdd_kernel<<<gridSize, blockSize>>>(a, b, c, n);
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
}

void my::cuda::matrixMultiplyShared(dim3 gridSize, dim3 blockSize, cudaStream_t stream, float *a, float *b, float *c, int n)
{
    matrixMultiplyShared_kernel<<<gridSize, blockSize, 0, stream>>>(a, b, c, n);
}

extern "C" void matrixMultiplyShared(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, int n)
{
    matrixMultiplyShared_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

__global__ void sharedABMultiply_kernel(float *a, float *b, float *c, int N)
{
    __shared__ float aTile[BLOCK_SIZE][BLOCK_SIZE], bTile[BLOCK_SIZE][BLOCK_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row * BLOCK_SIZE + threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y * N + col];
    __syncthreads();
    for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
        sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
    }
    c[row * N + col] = sum;
}

void my::cuda::sharedABMultiply(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, int n)
{
    sharedABMultiply_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

void my::cuda::sharedABMultiply(dim3 gridSize, dim3 blockSize, cudaStream_t stream, float *a, float *b, float *c, int n)
{
    sharedABMultiply_kernel<<<gridSize, blockSize, 0, stream>>>(a, b, c, n);
}

extern "C" void sharedABMultiply(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, int n)
{
    sharedABMultiply_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

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

void my::cuda::matrixMultiplyShared(
    dim3 gridSize, dim3 blockSize, cudaStream_t stream, float *a, float *b, float *c, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
{
    matrixMultiplyShared_kernel<<<gridSize, blockSize, 0, stream>>>(a, b, c, ARows, ACols, BRows, BCols, CRows, CCols);
}

/*
extern "C" void sharedABMultiply(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
{
    matrixMultiplyShared_kernel<<<gridSize, blockSize>>>(a, b, c, ARows, ACols, BRows, BCols, CRows, CCols);
    // cudaStreamSynchronize(0);
}*/

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
}

/**
 * \brief To fill 2D flat Matrix with value
 * \param int* matA 2D flat matrix
 * \param int value
 * \param size_t sizeAX size of matrix
 * \param size_t sizeAY size of matrix
 */
__global__ void matFill_kernel(int *matA, int value, size_t sizeAX, size_t sizeAY)
{
#warning "To do: test my::cuda::matFill_kernel"
    size_t col = threadIdx.x + blockIdx.x * blockDim.x;
    size_t row = threadIdx.y + blockIdx.y * blockDim.y;
    size_t index = row * sizeAX + col;
    if (col < sizeAX && row < sizeAY) {
        matA[index] = value;
    }
}

void my::cuda::matFill(dim3 gridSize, dim3 blockSize, int *matA, int value, size_t sizeAX, size_t sizeAY)
{
    matFill_kernel<<<gridSize, blockSize>>>(matA, value, sizeAX, sizeAY);
}

void my::cuda::matFill(dim3 gridSize, dim3 blockSize, cudaStream_t stream, int *matA, int value, size_t sizeAX, size_t sizeAY)
{
    matFill_kernel<<<gridSize, blockSize, 0, stream>>>(matA, value, sizeAX, sizeAY);
}

extern "C" void matFill(dim3 gridSize, dim3 blockSize, int *matA, int value, size_t sizeAX, size_t sizeAY)
{
    matFill_kernel<<<gridSize, blockSize>>>(matA, value, sizeAX, sizeAY);
}

/**
 * \brief To copy 2D flat Matrix to other matrix
 * \param int* matA 2D flat matrix
 * \param int* matB 2D flat matrix
 * \param size_t sizeAX size of matrix
 * \param size_t sizeAY size of matrix
 */
__global__ void matCopy_kernel(int *matB, int *matA, int width, int height)
{
#warning "To do: test my::cuda::matCopy_kernel"
    __shared__ int block[BLOCK_SIZE][BLOCK_SIZE + 1];

    unsigned int xIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    if ((xIndex < width) && (yIndex < height)) {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = matA[index_in];
    }
    __syncthreads();
    xIndex = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    if ((xIndex < height) && (yIndex < width)) {
        unsigned int index_out = yIndex * height + xIndex;
        matB[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

void my::cuda::matCopy(dim3 gridSize, dim3 blockSize, int *matA, int *matB, size_t sizeAX, size_t sizeAY)
{
    matCopy_kernel<<<gridSize, blockSize>>>(matA, matB, sizeAX, sizeAY);
}

void my::cuda::matCopy(dim3 gridSize, dim3 blockSize, cudaStream_t stream, int *matA, int *matB, size_t sizeAX, size_t sizeAY)
{
    matCopy_kernel<<<gridSize, blockSize, 0, stream>>>(matA, matB, sizeAX, sizeAY);
}

extern "C" void matCopy(dim3 gridSize, dim3 blockSize, int *matA, int *matB, size_t sizeAX, size_t sizeAY)
{
    matCopy_kernel<<<gridSize, blockSize>>>(matA, matB, sizeAX, sizeAY);
}