//https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/
//https://stackoverflow.com/questions/13553015/cuda-c-linker-error-undefined-reference


extern "C" {
// Can be remove if use "external *Function*" in .c/cpp file
#include <cuda_runtime.h>
#include "kernel.h"
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd_kernel(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}


__global__ void vecSub_kernel(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] - b[id];
}

__global__ void vecMult_kernel(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] - b[id];
}

__global__ void vecDiv_kernel(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] - b[id];
}


//extern "C"
void vecAdd(size_t gridSize, size_t blockSize, double *a, double *b, double *c, int n)
{
    vecAdd_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}


//extern "C"
void vecSub(size_t gridSize, size_t blockSize, double *a, double *b, double *c, int n)
{
    vecSub_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

//extern "C"
void vecMult(size_t gridSize, size_t blockSize, double *a, double *b, double *c, int n)
{
    vecMult_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

//extern "C"
void vecDiv(size_t gridSize, size_t blockSize, double *a, double *b, double *c, int n)
{
    vecDiv_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}






/*
// kernel.cu

__global__ void kernel()
{
    // some code here...
}

void kernel_function()
{

    dim3 threads(2, 1);
    dim3 blocks(1, 1);

    kernel<<<blocks, threads>>>();
}



//main.cpp

extern void kernel_function();

int main(int argc, char *argv[]){

  // some logic here...

  kernel_function();
  return 0;
} 
*/