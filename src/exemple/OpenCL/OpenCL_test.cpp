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
//  Created: 05, March, 2021                                //
//  Modified: 05, March, 2021                               //
//  file: OpenCL_test.cpp                                   //
//  Crypto                                                  //
//  Source: https://stackoverflow.com/questions/23992369/what-should-i-use-instead-of-clkernelfunctor                                                 //
//          https://github.com/Dakkers/OpenCL-examples                                                 //
//          https://github.com/Sable/Ostrich/blob/master/combinational-logic/crc/opencl/crc_kernel.cl                                                 //
//          https://www.cl.cam.ac.uk/teaching/1819/AdvGraphIP/03_OpenCL.pdf                                                 //
//          https://github.com/Sable/Ostrich/tree/master/combinational-logic/crc/opencl                                                 //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

//#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstack-protector"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wstrict-overflow"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#pragma GCC diagnostic ignored "-Wundef"
#if defined(__APPLE__) || defined(__MACOSX)
#    include <OpenCL/cl.cpp>
#else
#    include <CL/cl.hpp>
#endif
#pragma GCC diagnostic pop
#include <fstream>
#include <iostream>
extern "C"
{
#include <stdio.h>
}

/**
 * @brief 
 * 
 * @example OpenCL_test.cpp
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char **argv)
{
    std::string platform_ver;
    // get all platforms (drivers), e.g. NVIDIA
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform = all_platforms[0];

    // Get Platform version
    all_platforms[0].getInfo(CL_PLATFORM_VERSION, &platform_ver);

    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    std::cout << "Platform version: " << platform_ver << "\n";

    // get default device (CPUs, GPUs) of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    // use device[1] because that's a GPU; device[0] is the CPU
    cl::Device default_device = all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
    // a context is like a "runtime link" to the device and platform;
    // i.e. communication is possible
    cl::Context context({default_device});

    // create the program that we want to execute on the device
    cl::Program::Sources sources;

    /*
    // calculates for each element; C = A + B
    std::string kernel_code=
        "   void kernel simple_add(global const int* A, global const int* B, global int* C, "
        "                          global const int* N) {"
        "       int ID, Nthreads, n, ratio, start, stop;"
        ""
        "       ID = get_global_id(0);"
        "       Nthreads = get_global_size(0);"
        "       n = N[0];"
        ""
        "       ratio = (n / Nthreads);"  // number of elements for each thread
        "       start = ratio * ID;"
        "       stop  = ratio * (ID + 1);"
        ""
        "       for (int i=start; i<stop; i++)"
        "           C[i] = A[i] + B[i];"
        "   }";
    sources.push_back({kernel_code.c_str(), kernel_code.length()});*/

    // Read the program source
    std::ifstream sourceFile("../kernels/simple_add.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));

    cl::Program program(context, source);
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
        exit(1);
    }

    // apparently OpenCL only likes arrays ...
    // N holds the number of elements in the vectors we want to add

    int N[1] = {100};
    unsigned int n = N[0];

    // create buffers on device (allocate space on GPU)
    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_N(context, CL_MEM_READ_ONLY, sizeof(int));

    // create things on here (CPU)
    /*
    int A[n], B[n];
    for (int i = 0; i < n; i++) {
        A[i] = i;
        B[i] = n - i - 1;
    }*/
    int A[n], B[n];
    for (int i = 0; i < n; i++) {
        A[i] = 1;
        B[i] = i - 1;
    }

    // create a queue (a queue of commands that the GPU will execute)
    cl::CommandQueue queue(context, default_device);

    // push write commands to queue
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * n, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * n, B);
    queue.enqueueWriteBuffer(buffer_N, CL_TRUE, 0, sizeof(int), N);

    // RUN ZE KERNEL
    cl::Kernel simple_add(program, "simple_add");
    simple_add.setArg(0, buffer_A);
    simple_add.setArg(1, buffer_B);
    simple_add.setArg(2, buffer_C);
    simple_add.setArg(3, buffer_N);
    // queue.enqueueNDRangeKernel(simple_add, cl::NullRange, cl::NDRange(10), cl::NullRange);
    queue.enqueueNDRangeKernel(simple_add, cl::NullRange, cl::NDRange(100), cl::NullRange);
    // N_ELEMENTS * sizeof(int)
    int C[n];
    // read result from GPU to here
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * n, C);
    queue.finish();
    std::cout << "result: {";
    for (int i = 0; i < n; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << "}" << std::endl;

    return (EXIT_SUCCESS);
}