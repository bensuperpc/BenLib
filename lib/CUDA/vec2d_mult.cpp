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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.hpp"
extern "C"
{
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

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
    #pragma omp parallel for collapse(2)
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

    printf("Number of threads: %i (%ix%i)\n", Block_dim.x * Block_dim.y, Block_dim.x, Block_dim.y);
    printf("Number of blocks: %i (%ix%i)\n", Grid_dim.x * Grid_dim.y, Grid_dim.x, Grid_dim.y);

    // commented out the functions which helps to calculate time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // kernel call
    // multiply < < Grid_dim, Block_dim >> > (Left_Vector_d, Right_Vector_d, Res_d, dim);
    my::cuda::matrixMultiplyShared(Grid_dim, Block_dim, Left_Vector_d, Right_Vector_d, Res_d, dim);

    // commented out the functions which helps to calculate time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float et;
    cudaEventElapsedTime(&et, start, stop);

    // Retrieve result from device and store it in host array
    cudaMemcpy(Res_h, Res_d, vector_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(start, 0);

    cpu_matrix_mult(Left_Vector_h, Right_Vector_h, CPU, dim); // matrix multiplication on cpu

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float time_spent;
    cudaEventElapsedTime(&time_spent, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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