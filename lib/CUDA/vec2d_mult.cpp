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
//          https://github.com/kberkay/Cuda-Matrix-Multiplication/blob/master/matrix_Multiplication.cu                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.hpp"
#include "time.h"

#define BLOCK_SIZE 16

/*
 * prints matrices
 * Because matrices filled with dummy 0s function takes 3 dim arguments:
 *      actual x and y dimension and dim as big square matrix's dimension
 */
void print_matrices(float *matrix, char *file_Name, int x_dim, int y_dim, int dim)
{
    std::ofstream outFile;
    outFile.open(file_Name);

    outFile << std::fixed;
    outFile << std::setprecision(2);

    for (int i = 0; i < x_dim; i++) {

        for (int j = 0; j < y_dim; j++) {
            outFile << matrix[i * dim + j] << " ";
        }
        outFile << std::endl;
    }
}

// naive CPU matrix multiplication code
// because of its simplicity directly taken from web
// it multiplies square matrices
/*__host__*/ void cpu_matrix_mult(float *h_a, float *h_b, float *h_result, int m)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            float tmp = 0.0;
            for (int h = 0; h < m; ++h) {
                tmp += h_a[i * m + h] * h_b[h * m + j];
            }
            h_result[i * m + j] = tmp;
        }
    }
}

// this function is for filling the matrices with cos and sin values randomly
// I transform the matrices to square matrix in order to perform better multiplication
/*__host__*/ int fill(float **Lmatrix, float **Rmatrix, int LdimX, int LdimY, int RdimX, int RdimY)
{

    int sqr_dim_X, sqr_dim_Y, size;

    sqr_dim_X = RdimX;
    if (LdimX > RdimX) {
        sqr_dim_X = LdimX;
    }

    sqr_dim_Y = RdimY;
    if (LdimY > RdimY) {
        sqr_dim_Y = LdimY;
    }

    size = sqr_dim_Y;
    if (sqr_dim_X > sqr_dim_Y) {
        size = sqr_dim_X;
    }

    int temp = size / BLOCK_SIZE + (size % BLOCK_SIZE == 0 ? 0 : 1);
    size = temp * BLOCK_SIZE;

    size_t pt_size = size * size * sizeof(float);

    *Lmatrix = (float *)malloc(pt_size);
    *Rmatrix = (float *)malloc(pt_size);

    memset(*Lmatrix, 0, pt_size);
    memset(*Rmatrix, 0, pt_size);

    for (int i = 0; i < LdimX; i++) {
        for (int j = 0; j < LdimY; j++) {
            int dummy = size * i + j;
            (*Lmatrix)[dummy] = sinf(dummy);
        }
    }
    for (int i = 0; i < RdimX; i++) {
        for (int j = 0; j < RdimY; j++) {
            int dummy = size * i + j;
            (*Rmatrix)[dummy] = cosf(dummy);
        }
    }
    return size;
}

// main routine that executes on the host
int main(void)
{
    // size of the vectors to be processed  and matrix dimensions
    int Left_matrix_x, Left_matrix_y, Right_matrix_x, Right_matrix_y, Left_vector_size, Right_vector_size;

    float *Left_Vector_h, *Right_Vector_h, *Left_Vector_d, *Right_Vector_d, *Res_h, *Res_d, *CPU; // Pointer to host & device arrays

    printf("Enter m n n k :\n");

    scanf("%d %d %d %d", &Left_matrix_x, &Left_matrix_y, &Right_matrix_x, &Right_matrix_y); // input matrix dimensions are taken

    int dim = fill(&Left_Vector_h, &Right_Vector_h, Left_matrix_x, Left_matrix_y, Right_matrix_x, Right_matrix_y); // fills the matrices with random values

    // print_matrices(Left_Vector_h,"Input_LHS",Left_matrix_x,Left_matrix_y,dim);
    // print_matrices(Right_Vector_h,"Input_RHS",Right_matrix_x,Right_matrix_y,dim);

    size_t vector_size;
    vector_size = dim * dim * sizeof(float);

    Res_h = (float *)malloc(vector_size); // Allocate array on host for result
    CPU = (float *)malloc(vector_size);   // Allocate array on host for CPU_matrix_multiplication result

    cudaMalloc((void **)&Left_Vector_d, vector_size);  // Allocate array on device for LHS operand
    cudaMalloc((void **)&Right_Vector_d, vector_size); // Allocate array on device for RHS operand but this is vector 1xN
    cudaMalloc((void **)&Res_d, vector_size);          // Allocate array on device for result

    cudaMemcpy(Left_Vector_d, Left_Vector_h, vector_size, cudaMemcpyHostToDevice);   // copy values to device
    cudaMemcpy(Right_Vector_d, Right_Vector_h, vector_size, cudaMemcpyHostToDevice); // copy values to device

    // Block dimension is directly from block_size
    dim3 Block_dim(BLOCK_SIZE, BLOCK_SIZE);
    // Grid dimension is found by dividing matrix dimension to block_size
    dim3 Grid_dim(dim / BLOCK_SIZE, dim / BLOCK_SIZE);

    // commented out the functions which helps to calculate time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // kernel call
    // multiply < < Grid_dim, Block_dim >> > (Left_Vector_d, Right_Vector_d, Res_d, dim);
    my::cuda::multiply(Grid_dim, Block_dim, Left_Vector_d, Right_Vector_d, Res_d, dim);

    // commented out the functions which helps to calculate time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float et;
    cudaEventElapsedTime(&et, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Retrieve result from device and store it in host array
    cudaMemcpy(Res_h, Res_d, vector_size, cudaMemcpyDeviceToHost);

    clock_t begin = clock();

    cpu_matrix_mult(Left_Vector_h, Right_Vector_h, CPU, dim); // matrix multiplication on cpu

    clock_t end = clock();
    double time_spent = (double)1000 * (end - begin) / CLOCKS_PER_SEC;

    // commented out the functions which helps to calculate time
    printf("GPU time= %f ms\n", et);

    printf("CPU time= %lf ms\n", time_spent);

    // Prints the results
    // print_matrices(Res_h,"GPU_out",Left_matrix_x,Right_matrix_y,dim);
    // print_matrices(CPU,"CPU_out",Left_matrix_x,Right_matrix_y,dim);

    bool eqaul = true;
    for (int i = 0; i < Left_matrix_x && eqaul; i++) {
        for (int j = 0; j < Right_matrix_y && eqaul; j++) {
            if (abs(Res_h[i * dim + j] - CPU[i * dim + j]) > 0.001) {
                eqaul = false;
                printf("NOT EQUAL\n");
            }
        }
    }
    if (eqaul) {
        std::cout << "Results are equal!" << std::endl;
    } else {
        std::cout << "Results are NOT equal!" << std::endl;
    }

    // Cleanup
    free(Left_Vector_h);
    free(Right_Vector_h);
    free(Res_h);
    free(CPU);
    cudaFree(Left_Vector_d);
    cudaFree(Right_Vector_d);
    cudaFree(Res_d);
}

/*
const int TILE_WIDTH = 16;
#define THREADS_PER_BLOCK 1024

void matrixMultiplyCPU(float *a, float *b, float *c, int width)
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

void matrixMultiplyCPU_MP(float *a, float *b, float *c, int width)
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
    int width = 16; // Define width of square matrix
    // Initialise grid and block variables
    int sqrtThreads = sqrt(THREADS_PER_BLOCK);
    int nBlocks = width / sqrtThreads;
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

    int size; // Number of bytes required by arrays

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

    dim3 Block_dim = {TILE_WIDTH, TILE_WIDTH};
    dim3 Grid_dim = {width / TILE_WIDTH, width / TILE_WIDTH};

    matrixMultiplyOptimised(Block_dim, , a_d, b_d, c_d, width);
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
            if(i > 0)
                printf("c_h: %f, d_h: %f, i: %li\n", c_h[i - 1], d_h[i - 1], i - 1);
            printf("Error: CPU and GPU results do not match\n");
            printf("c_h: %f, d_h: %f, i: %li\n", c_h[i], d_h[i], i);
        }
    }

    // Start timer for GPU (optimised)
    cudaEventRecord(start, 0);

    // Launch kernel (optimised)
    // matrixMultiplyOptimised<<<grid, block>>>(a_h, b_h, c_h, width);
    matrixMultiplySimple(grid, block, a_d, b_d, c_d, width);

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
*/