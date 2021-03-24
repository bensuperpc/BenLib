
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
//  Created: 24, March, 2021                                //
//  Modified: 24, March, 2021                               //
//  file: kernel.h                                          //
//  Crypto                                                  //
//  Source:                                                 //
//
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <cuda_utils.hpp>
extern "C"
{
#include <cuda_utils.h>
}

extern "C"
{
    void device()
    {
        my::cuda::device();
    }
}

void my::cuda::device()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (nDevices == 0) {
        printf("Cuda device not found.\n");
        return;
    }

    printf("Found %i Cuda device(s).\n", nDevices);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1000);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  totalGlobablMem (GB): %lu\n", (unsigned long)prop.totalGlobalMem);
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

extern "C"
{
    void driver()
    {
        my::cuda::driver();
    }
}

void my::cuda::driver()
{
    int driverVersion = 0;
    int runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Driver Version / Runtime Version: %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000,
        (runtimeVersion % 100) / 10);
}