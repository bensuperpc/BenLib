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
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matrix.hpp"
#include "matrix.tpp"

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
template <typename T> void print_matrices(T *matrix, char *file_Name, T x_dim, size_t y_dim, size_t dim)
{
    std::ofstream outFile;
    outFile.open(file_Name);

    outFile << std::fixed;
    outFile << std::setprecision(3);

    for (size_t i = 0; i < x_dim; i++) {
        for (size_t j = 0; j < y_dim; j++) {
            outFile << matrix[i * dim + j] << " ";
        }
        outFile << std::endl;
    }
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
                (*Rmatrix)[dummy] = (T)rand_r(dummy);
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

void device()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if(nDevices == 0) {
        printf("Cuda device not found.\n");
        return;
    }

    printf("Found %i Cuda device(s).\n",nDevices);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  totalGlobablMem: %lu\n", (unsigned long)prop.totalGlobalMem);
        printf("  sharedMemPerBlock: %i\n", prop.sharedMemPerBlock);
        printf("  regsPerBlock: %i\n", prop.regsPerBlock);
        printf("  warpSize: %i\n", prop.warpSize);
        printf("  memPitch: %i\n", prop.memPitch);
        printf("  maxThreadsPerBlock: %i\n", prop.maxThreadsPerBlock);
        printf("  maxThreadsDim: %i, %i, %i\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  maxGridSize: %i, %i, %i\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  clockRate (MHz): %i\n", prop.clockRate / 1000);
        printf("  totalConstMem: %i\n", prop.totalConstMem);
        printf("  major: %i\n", prop.major);
        printf("  minor: %i\n", prop.minor);
        printf("  textureAlignment: %i\n", prop.textureAlignment);
        printf("  deviceOverlap: %i\n", prop.deviceOverlap);
        printf("  multiProcessorCount: %i\n", prop.multiProcessorCount);
    }
}

void driver()
{
    int driverVersion = 0;
    int runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Driver Version / Runtime Version: %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000,
        (runtimeVersion % 100) / 10);
}

// main routine that executes on the host
int main(void)
{
    cudaSetDevice(0);
    // cudaDeviceEnablePeerAccess(0, 0);
    device();
    driver();

    int priority_high, priority_low;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    cudaStream_t st_high, st_low;
    cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high);
    cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);
    cudaStream_t stream;
    // cudaStreamAttachMemAsync(stream1, &x);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // size of the vectors to be processed  and matrix dimensions
    int Left_matrix_x, Left_matrix_y, Right_matrix_x, Right_matrix_y, Left_vector_size, Right_vector_size;

    float *Left_Vector_h, *Right_Vector_h, *Left_Vector_d, *Right_Vector_d, *Res_h, *Res_d, *CPU; // Pointer to host & device arrays

    printf("Enter m n n k :\n");

    scanf("%d %d %d %d", &Left_matrix_x, &Left_matrix_y, &Right_matrix_x, &Right_matrix_y); // input matrix dimensions are taken

    // return dimention of Matc to store result
    int dim = allocFill<float>(
        &Left_Vector_h, &Right_Vector_h, Left_matrix_x, Left_matrix_y, Right_matrix_x, Right_matrix_y, true); // fills the matrices with random values

    print_matrices<float>(Left_Vector_h, "Input_LHS", Left_matrix_x, Left_matrix_y, dim);
    print_matrices<float>(Right_Vector_h, "Input_RHS", Right_matrix_x, Right_matrix_y, dim);

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
}