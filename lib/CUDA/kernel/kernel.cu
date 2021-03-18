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
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

extern "C"
{
// Can be remove if use "external *Function*" in .c/cpp file
#include "kernel.h"
}
#include "kernel.cuh"
#include "kernel.hpp"

__global__ void vecAdd_kernel(double *a, double *b, double *c, size_t n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

void my::cuda::vecAdd(size_t gridSize, size_t blockSize, double *a, double *b, double *c, size_t n)
{
    vecAdd_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

extern "C" void vecAdd(size_t gridSize, size_t blockSize, double *a, double *b, double *c, size_t n)
{
    vecAdd_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

__global__ void vecSub_kernel(double *a, double *b, double *c, size_t n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] - b[id];
}

void my::cuda::vecSub(size_t gridSize, size_t blockSize, double *a, double *b, double *c, size_t n)
{
    vecSub_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

// extern "C"
void vecSub(size_t gridSize, size_t blockSize, double *a, double *b, double *c, size_t n)
{
    vecSub_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

__global__ void vecMult_kernel(double *a, double *b, double *c, size_t n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] - b[id];
}

void my::cuda::vecMult(size_t gridSize, size_t blockSize, double *a, double *b, double *c, size_t n)
{
    vecMult_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

extern "C" void vecMult(size_t gridSize, size_t blockSize, double *a, double *b, double *c, size_t n)
{
    vecMult_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

__global__ void vecDiv_kernel(double *a, double *b, double *c, size_t n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] - b[id];
}

void my::cuda::vecDiv(size_t gridSize, size_t blockSize, double *a, double *b, double *c, size_t n)
{
    vecDiv_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

extern "C" void vecDiv(size_t gridSize, size_t blockSize, double *a, double *b, double *c, size_t n)
{
    vecDiv_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

__global__ void matrixMultiplySimple_kernel(float *a, float *b, float *c, size_t width)
{
    size_t col = threadIdx.x + blockIdx.x * blockDim.x;
    size_t row = threadIdx.y + blockIdx.y * blockDim.y;

    float result = 0;

    if (col < width && row < width) {
        for (size_t k = 0; k < width; k++) {
            result += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = result;
    }
}

void my::cuda::matrixMultiplySimple(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, size_t n)
{
    matrixMultiplySimple_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

extern "C" void matrixMultiplySimple(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, size_t n)
{
    matrixMultiplySimple_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

const int TILE_WIDTH = 16;
__global__ void matrixMultiplyOptimised_kernel(float *a, float *b, float *c, size_t width)
{
    // Allocate 2D tiles in shared memory
    __shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_b[TILE_WIDTH][TILE_WIDTH];

    // Calculate row and column index of element
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    float result = 0;

    // Loop over tiles of input in phases
    for (size_t p = 0; p < width / TILE_WIDTH; p++) {
        // Collaboratively load tiles into shared memory
        s_a[threadIdx.y][threadIdx.x] = a[row * width + (p * TILE_WIDTH + threadIdx.x)];
        s_b[threadIdx.y][threadIdx.x] = b[(p * TILE_WIDTH + threadIdx.y) * width + col];

        // Wait until all data is loaded before allowing any threads in the block to continue
        __syncthreads();

        // Dot product between row of s_a and column of s_b
        for (size_t i = 0; i < TILE_WIDTH; i++) {
            result += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
        }

        // Wait until all calculations are finished before allowing any threads in the block to continue
        __syncthreads();
    }

    // Write result
    c[row * width + col] = result;
}

void my::cuda::matrixMultiplyOptimised(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, size_t n)
{
    matrixMultiplyOptimised_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

extern "C" void matrixMultiplyOptimised(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, size_t n)
{
    matrixMultiplyOptimised_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

__global__ void matrixAddKernel_kernel(int *a, int *b, int *c, size_t N)
{
    size_t col = threadIdx.x + blockIdx.x * blockDim.x;
    size_t row = threadIdx.y + blockIdx.y * blockDim.y;
    size_t index = row * N + col;
    if (col < N && row < N) {
        c[index] = a[index] + b[index];
        // printf("Hello from blockx %d, threadx %d\n", blockIdx.x, threadIdx.x);
        // printf("Hello from blocky %d, thready %d\n", blockIdx.y, threadIdx.y);
        // printf("Hello from col %d, row %d\n", col, row);
        // printf("c:%d = a:%d + b:%d\n", c[index], a[index], b[index]);
    }
    //__syncthreads();
}

void my::cuda::matrixAddKernel(dim3 gridSize, dim3 blockSize, int *a, int *b, int *c, size_t n)
{
    matrixAddKernel_kernel<<<gridSize, blockSize>>>(a, b, c, n);
    cudaStreamSynchronize(0);
    //cudaDeviceSynchronize();
}

/*
void my::cuda::matrixAddKernel(dim3 gridSize, dim3 blockSize, cudaStream_t *streams, int *a, int *b, int *c, size_t n)
{
    matrixAddKernel_kernel<<<gridSize, blockSize, 0, streams>>>(a, b, c, n);
}
*/

extern "C" void matrixAddKernel(dim3 gridSize, dim3 blockSize, int *a, int *b, int *c, size_t n)
{
    matrixAddKernel_kernel<<<gridSize, blockSize>>>(a, b, c, n);
    cudaStreamSynchronize(0);
    //cudaDeviceSynchronize();
}

/*
extern "C" void matrixAddKernelS(dim3 gridSize, dim3 blockSize, cudaStream_t *streams, int *a, int *b, int *c, size_t n)
{
    matrixAddKernel_kernel<<<gridSize, blockSize, 0, streams>>>(a, b, c, n);
}
*/

__global__ void matrixMultiplyShared_kernel(float *left, float *right, float *res, int dim)
{

    int i, j;
    float temp = 0;

    __shared__ float Left_shared_t[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Right_shared_t[BLOCK_SIZE][BLOCK_SIZE];

    // Row i of matrix left
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {

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
        for (int k = 0; k < BLOCK_SIZE; k++) {

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
    cudaStreamSynchronize(0);
}

/*
void my::cuda::matrixMultiplyShared(dim3 gridSize, dim3 blockSize, cudaStream_t *streams, float *a, float *b, float *c, int n)
{
    matrixMultiplyShared_kernel<<<gridSize, blockSize, 0, streams>>>(a, b, c, n);
}
*/

extern "C" void matrixMultiplyShared(dim3 gridSize, dim3 blockSize, float *a, float *b, float *c, int n)
{
    matrixMultiplyShared_kernel<<<gridSize, blockSize>>>(a, b, c, n);
    cudaStreamSynchronize(0);
}


/*
extern "C" void matrixMultiplySharedS(dim3 gridSize, dim3 blockSize, cudaStream_t *streams, float *a, float *b, float *c, int n)
{
    matrixMultiplyShared_kernel<<<gridSize, blockSize, 0, streams>>>(a, b, c, n);
}
*/

__global__ void sharedABMultiply(float *a, float* b, float *c,
                                 int N)
{
    __shared__ float aTile[BLOCK_SIZE][BLOCK_SIZE],
                     bTile[BLOCK_SIZE][BLOCK_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*BLOCK_SIZE+threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*N+col];
    __syncthreads();
    for (int i = 0; i < BLOCK_SIZE; i++) {
        sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
    }
    c[row*N+col] = sum;
}