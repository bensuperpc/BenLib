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
//          https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#optimize
//          https://medium.com/gpgpu/multi-gpu-programming-6768eeb42e2c
//          https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/
//          https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
//          https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
//          https://github.com/evlasblom/cuda-opencv-examples
//          https://github.com/NVIDIA/cuda-samples/blob/master/Samples/matrixMul/matrixMul.cu
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include "cuda_runtime.h"
#include "cuda_utils.hpp"
#include "device_launch_parameters.h"
#include "matrix.tpp"
#include "matrix_ops.tpp"

extern "C"
{
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

template <typename T>
/*__host__*/ int allocFill(T **Lmatrix, T **Rmatrix, size_t LdimX, size_t LdimY, size_t RdimX, size_t RdimY, bool Unified_memory = false)
{
    size_t sqr_dim_X, sqr_dim_Y, size;

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

    size_t temp = size / BLOCK_SIZE + (size % BLOCK_SIZE == 0 ? 0 : 1);
    size = temp * BLOCK_SIZE;

    size_t pt_size = size * size * sizeof(T);

    // If Unified memory is enable
    if (Unified_memory == false) {
        *Lmatrix = (T *)malloc(pt_size);
        *Rmatrix = (T *)malloc(pt_size);
        // cudaMallocHost((void**)Lmatrix, pt_size);
        // cudaMallocHost((void**)Rmatrix, pt_size);
    } else {
        cudaMallocManaged(Lmatrix, pt_size); // cudaMemAttachHost
        cudaMallocManaged(Rmatrix, pt_size); // cudaMemAttachHost
    }

    // set all value to 0
    memset(*Lmatrix, 0, pt_size);
    memset(*Rmatrix, 0, pt_size);

#pragma omp parallel for collapse(2) schedule(auto)
    for (size_t i = 0; i < LdimX; i++) {
        for (size_t j = 0; j < LdimY; j++) {
            size_t dummy = size * i + j;
            if constexpr (std::is_floating_point_v<T>) {
                (*Lmatrix)[dummy] = (T)sinf(dummy);
            } else {
                (*Lmatrix)[dummy] = (T)rand_r(dummy);
            }
        }
    }

#pragma omp parallel for collapse(2) schedule(auto)
    for (size_t i = 0; i < RdimX; i++) {
        for (size_t j = 0; j < RdimY; j++) {
            size_t dummy = size * i + j;
            if constexpr (std::is_floating_point_v<T>) {
                (*Rmatrix)[dummy] = (T)cosf(dummy);
            } else {
                (*Rmatrix)[dummy] = (T)rand_r(dummy);
            }
        }
    }
    return size;
}

template <typename T>
/*__host__*/ void matAllocFill(T **matA, T **matB, T **matC, dim3 &dimsA, dim3 &dimsB, dim3 &dimsC, bool Unified_memory = true, bool Pinned_memory = false)
{

    if (Unified_memory == true && Pinned_memory == true) {
    }

    size_t size_A = dimsA.x * dimsA.y;
    size_t mem_size_A = sizeof(T) * size_A;

    size_t size_B = dimsB.x * dimsB.y;
    size_t mem_size_B = sizeof(T) * size_B;

    dimsC = dim3(dimsB.x, dimsA.y, 1);
    size_t mem_size_C = dimsC.x * dimsC.y * sizeof(T);

    // If Unified memory is enable
    if (Unified_memory == true) {
        gpuErrchk(cudaMallocManaged(reinterpret_cast<void **>(matA), mem_size_A)); // Unified memory
        gpuErrchk(cudaMallocManaged(reinterpret_cast<void **>(matB), mem_size_B)); // Unified memory
        gpuErrchk(cudaMallocManaged(reinterpret_cast<void **>(matC), mem_size_C)); // Unified memory
    } else if (Pinned_memory == true) {
#ifdef __CUDACC__ || __CUDA_ARCH__
        gpuErrchk(cudaMallocHost(reinterpret_cast<void **>(matA), mem_size_A)); // host pinned
        gpuErrchk(cudaMallocHost(reinterpret_cast<void **>(matB), mem_size_B)); // host pinned
        gpuErrchk(CudaMallocHost(reinterpret_cast<void **>(matC), mem_size_C)); // host pinned
#else
#    warning Use malloc instead CudaMallocHost
        *matA = (T *)malloc(mem_size_A); // host pageable
        *matB = (T *)malloc(mem_size_B); // host pageable
        *matC = (T *)malloc(mem_size_C); // host pageable
#endif
    } else {
        *matA = (T *)malloc(mem_size_A); // host pageable
        *matB = (T *)malloc(mem_size_B); // host pageable
        *matC = (T *)malloc(mem_size_C); // host pageable
    }

    if (*matA == NULL || *matB == NULL || *matC == NULL) {
        fprintf(stderr, "Failed to allocate matrix!\n");
        exit(EXIT_FAILURE);
    }

    // set all value to 0
    memset(*matA, 0, mem_size_A);
    memset(*matB, 0, mem_size_B);
    memset(*matC, 0, mem_size_C);

#pragma omp parallel for collapse(2) schedule(auto)
    for (size_t i = 0; i < dimsA.x; i++) {
        for (size_t j = 0; j < dimsA.y; j++) {
            size_t dummy = dimsA.x * i + j;
            if constexpr (std::is_floating_point_v<T>) {
                (*matA)[dummy] = (T)sinf(dummy);
            } else {
                (*matA)[dummy] = (T)rand_r(dummy);
            }
        }
    }

#pragma omp parallel for collapse(2) schedule(auto)
    for (size_t i = 0; i < dimsB.x; i++) {
        for (size_t j = 0; j < dimsB.x; j++) {
            size_t dummy = dimsB.x * i + j;
            if constexpr (std::is_floating_point_v<T>) {
                (*matB)[dummy] = (T)cosf(dummy);
            } else {
                (*matB)[dummy] = (T)rand_r(dummy);
            }
        }
    }
}

// main routine that executes on the host
int main(void)
{
    cudaSetDevice(0);
    // cudaDeviceEnablePeerAccess(0, 0);
    my::cuda::device();
    my::cuda::driver();

    int priority_high, priority_low;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    cudaStream_t st_high, st_low;
    cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high);
    cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);
    cudaStream_t stream;
    // cudaStreamAttachMemAsync(stream1, &x);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    printf("Enter m n n k :\n");

    dim3 dimsA(5 * 2 * BLOCK_SIZE, 5 * 2 * BLOCK_SIZE, 1);
    dim3 dimsB(5 * 4 * BLOCK_SIZE, 5 * 2 * BLOCK_SIZE, 1);
    dim3 dimsC(5 * 4 * BLOCK_SIZE, 5 * 2 * BLOCK_SIZE, 1);

    scanf("%d %d %d %d", &dimsA.x, &dimsA.y, &dimsB.x, &dimsB.y);

    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n", dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    float *MatA, *MatB, *MatC;

    matAllocFill<float>(&MatA, &MatB, &MatC, dimsA, dimsB, dimsC);

    cudaMemPrefetchAsync(MatA, dimsA.x * dimsA.y * sizeof(float), 0, stream);
    cudaMemPrefetchAsync(MatB, dimsB.x * dimsB.y * sizeof(float), 0, stream);

    printf("MatrixA(%d,%d), MatrixB(%d,%d), MatrixC(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y, dimsC.x, dimsC.y);

    // Setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    my::cuda::MatrixMulCUDA(grid, threads, stream, MatA, MatB, MatC, dimsA.x, dimsB.x);
    cudaStreamSynchronize(stream);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    printf("GPU time: %f ms\n", gpu_time);
    printf("GPU perf: %f KOps/s\n", ((dimsC.x * dimsC.y) / gpu_time) / 1000.0);

    // my::cuda::print_matrices<float>(MatA, "test", dimsA.x, dimsA.y, dimsA.x);

    // my::cuda::matrixMultiplyShared(grid, threads, stream, MatA, MatB, MatB, 100);
    //<<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);

    /*
    int Left_matrix_x, Left_matrix_y, Right_matrix_x, Right_matrix_y, Left_vector_size, Right_vector_size;

    float *Left_Vector_h, *Right_Vector_h, *Left_Vector_d, *Right_Vector_d, *Res_h, *Res_d, *CPU; // Pointer to host & device arrays

    printf("Enter m n n k :\n");
    // scanf("%d %d %d %d", &Left_matrix_x, &Left_matrix_y, &Right_matrix_x, &Right_matrix_y); // input matrix dimensions are taken


    // return dimention of Matc to store result


    size_t vector_size;
    vector_size = dim * dim * sizeof(float);

    // Res_h = (float *)malloc(vector_size); // Allocate array on host for result
    CPU = (float *)malloc(vector_size); // Allocate array on host for CPU_matrix_multiplication result
    // cudaMallocHost((void **)&Res_h, vector_size);

    // gpuErrchk(cudaMalloc((void **)&Left_Vector_d, vector_size));  // Allocate array on device for LHS operand
    // gpuErrchk(cudaMalloc((void **)&Right_Vector_d, vector_size)); // Allocate array on device for RHS operand but this is vector 1xN
    // gpuErrchk(cudaMalloc((void **)&Res_d, vector_size));          // Allocate array on device for result
    cudaMallocManaged((void **)&Res_d, vector_size);

    // gpuErrchk(cudaMemcpyAsync(Left_Vector_d, Left_Vector_h, vector_size, cudaMemcpyHostToDevice, stream));   // copy values to device
    // gpuErrchk(cudaMemcpyAsync(Right_Vector_d, Right_Vector_h, vector_size, cudaMemcpyHostToDevice, stream)); // copy values to device
    cudaMemPrefetchAsync(Left_Vector_h, vector_size, 0, stream);
    cudaMemPrefetchAsync(Right_Vector_h, vector_size, 0, stream);

    // Block dimension is directly from block_size
    dim3 Block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
    // Grid dimension is found by dividing matrix dimension to block_size
    dim3 Grid_dim(dim / BLOCK_SIZE, dim / BLOCK_SIZE, 1);

    printf("Number of threads: %i (%ix%i)\n", Block_dim.x * Block_dim.y, Block_dim.x, Block_dim.y);
    printf("Number of blocks: %i (%ix%i)\n", Grid_dim.x * Grid_dim.y, Grid_dim.x, Grid_dim.y);
    printf("Output matrix size: %i (%ix%i)\n", dim * dim, dim, dim);
    size_t matrix_lenght = (dim * dim) * sizeof(float);
    if (matrix_lenght < 1000000) {
        printf("Matrix lenght: %f Ko (x3)\n", (double)((dim * dim) * sizeof(float)) / 1000.0);
    } else {
        printf("Matrix lenght: %f Mo (x3)\n", (double)((dim * dim) * sizeof(float)) / 1000000.0);
    }
    printf("\n");

    // commented out the functions which helps to calculate time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // kernel call
    // multiply < < Grid_dim, Block_dim >> > (Left_Vector_d, Right_Vector_d, Res_d, dim);

    // my::cuda::matrixMultiplyShared(Grid_dim, Block_dim, stream, Left_Vector_d, Right_Vector_d, Res_d, dim);
    // cudaMemPrefetchAsync(Res_d, vector_size, 0, stream);

    my::cuda::matrixMultiplyShared(Grid_dim, Block_dim, stream, Left_Vector_h, Right_Vector_h, Res_d, dim);

    // commented out the functions which helps to calculate time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Retrieve result from device and store it in host array
    // gpuErrchk(cudaMemcpyAsync(Res_h, Res_d, vector_size, cudaMemcpyDeviceToHost, stream));
    // cudaMemPrefetchAsync(Res_d, vector_size, 0, stream); // Optional for faster Unified memory
    // Block main thread until idle stream
    cudaStreamSynchronize(stream);
    // cudaStreamQuery(stream)

    cudaEventRecord(start, 0);

    my::cuda::matMultFlat<float>(Left_Vector_h, Right_Vector_h, CPU, dim); // matrix multiplication on cpu

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cpu_time;
    cudaEventElapsedTime(&cpu_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // commented out the functions which helps to calculate time

    printf("GPU time: %f ms\n", gpu_time);
    printf("GPU perf: %f KOps/s\n", ((dim * dim) / gpu_time) / 1000.0);

    printf("CPU time: %lf ms\n", cpu_time);
    printf("CPU perf: %f KOps/s\n", ((dim * dim) / cpu_time) / 1000.0);

    printf("CPU/GPU perf diff: x%lf\n", cpu_time / gpu_time);

    // Prints the results
    // print_matrices<float>(Res_h, "GPU_out", Left_matrix_x, Right_matrix_y, dim);
    print_matrices<float>(Res_d, "GPU_out", Left_matrix_x, Right_matrix_y, dim);
    print_matrices<float>(CPU, "CPU_out", Left_matrix_x, Right_matrix_y, dim);

    bool equal = true;
    for (int i = 0; i < Left_matrix_x && equal; i++) {
        for (int j = 0; j < Right_matrix_y && equal; j++) {
            // if (abs(Res_h[i * dim + j] - CPU[i * dim + j]) > 0.001) {
            if (abs(Res_d[i * dim + j] - CPU[i * dim + j]) > 0.001) {
                equal = false;
                printf("NOT EQUAL\n");
            }
        }
    }
    if (equal) {
        std::cout << "Results are equal!" << std::endl;
    } else {
        std::cout << "Results are NOT equal!" << std::endl;
    }

    // Cleanup
    // free(Left_Vector_h);
    // free(Right_Vector_h);
    // free(Res_h);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(st_high);
    cudaStreamDestroy(st_low);
    free(CPU);
    // cudaFreeHost(Res_h);

    // cudaFree(Left_Vector_d);
    // cudaFree(Right_Vector_d);
    cudaFree(Left_Vector_h);
    cudaFree(Right_Vector_h);
    cudaFree(Res_d);
    cudaDeviceReset();
    */
}