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
//  Source: https://stackoverflow.com/a/26518143/10152334                                                 //
//          https://stackoverflow.com/questions/48096034/a-simple-example-with-opencl                                                 //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <string>

#define __CL_ENABLE_EXCEPTIONS
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

#define KERNEL_FILE "../kernels/findStringInv.cl"
#define FUNCTION_NAME "findStringInv"
//__kernel void CRC32_1byte_tableless(__global const void *data, ulong length, uint previousCrc32, __global uint *resultCrc32)

int main(int argc, char **argv)
{
    // std::cout << crc32Lookup[0][1] << std::endl;
    const int N_ELEMENTS = 11;
    unsigned int platform_id = 0, device_id = 0;

    try {
        //std::unique_ptr<uint[]> A(new uint[N_ELEMENTS]);
        
        auto A = std::make_unique<std::array<unsigned char, N_ELEMENTS>>(std::array<unsigned char, N_ELEMENTS>{'H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd'});

        std::unique_ptr<uint> B(new uint);
        B = std::make_unique<uint>(N_ELEMENTS);
        //std::unique_ptr<int[]> C(new int[N_ELEMENTS]);
        std::unique_ptr<uint> C(new uint);
        C = std::make_unique<uint>(0);

        // Query for platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // Get a list of devices on this platform
        std::vector<cl::Device> devices;
        platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices); // Select the platform.
        
        // Create a context
        cl::Context context(devices);

        // Create a command queue
        cl::CommandQueue queue = cl::CommandQueue(context, devices[device_id]); // Select the device.

        // Create the memory buffers
        cl::Buffer bufferA = cl::Buffer(context, CL_MEM_WRITE_ONLY, N_ELEMENTS * sizeof(unsigned char));
        cl::Buffer bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, N_ELEMENTS * sizeof(uint));

        // Copy the input data to the input buffers using the command queue.
        queue.enqueueWriteBuffer(bufferA, CL_FALSE, 0, N_ELEMENTS * sizeof(unsigned char), A.get());
        queue.enqueueWriteBuffer(bufferC, CL_FALSE, 0, N_ELEMENTS * sizeof(uint), C.get());

        // Read the program source
        std::ifstream sourceFile(KERNEL_FILE);
        std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));

        // Make program from the source code
        cl::Program program = cl::Program(context, source);
        
        // Build the program for the devices
        program.build(devices);
        
        // Make kernel
        cl::Kernel vecadd_kernel(program, FUNCTION_NAME);
        
        // Set the kernel arguments
        vecadd_kernel.setArg(0, bufferA); // Data
        vecadd_kernel.setArg(1, (uint64_t)N_ELEMENTS); // lenght
        vecadd_kernel.setArg(2, 0); // Previous CRC
        vecadd_kernel.setArg(3, bufferC);

        // Execute the kernel
        cl::NDRange global(N_ELEMENTS);
        cl::NDRange local(256);
        
        queue.enqueueNDRangeKernel(vecadd_kernel, cl::NullRange, global, cl::NullRange);
        
        // Copy the output data back to the host
        std::cout << std::hex << "CRC OpenCL: 0x" << *C << std::endl;
    }
    catch (cl::Error err) {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        return (EXIT_FAILURE);
    }

    std::cout << "Done.\n";
    return (EXIT_SUCCESS);
}

/*
int main(int argc, char **argv)
{
    // std::cout << crc32Lookup[0][1] << std::endl;
    const int N_ELEMENTS = 11;
    unsigned int platform_id = 0, device_id = 0;

    try {
        //std::unique_ptr<uint[]> A(new uint[N_ELEMENTS]);

        auto A = std::make_unique<std::array<uint, 11>>(std::array<uint, 11>{72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100});

        std::unique_ptr<uint> B(new uint);
        B = std::make_unique<uint>(5);
        std::unique_ptr<int[]> D(new int[N_ELEMENTS]);

        std::unique_ptr<uint> C(new uint);
        C = std::make_unique<uint>(1);

        // Query for platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // Get a list of devices on this platform
        std::vector<cl::Device> devices;
        platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices); // Select the platform.

        // Create a context
        cl::Context context(devices);

        // Create a command queue
        cl::CommandQueue queue = cl::CommandQueue(context, devices[device_id]); // Select the device.

        // Create the memory buffers
        cl::Buffer bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, N_ELEMENTS * sizeof(uint));
        //cl::Buffer bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, 1 * sizeof(uint));
        //cl::Buffer bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, 1 * sizeof(uint));
        cl::Buffer bufferD = cl::Buffer(context, CL_MEM_WRITE_ONLY, N_ELEMENTS * sizeof(uint));

        // Copy the input data to the input buffers using the command queue.
        queue.enqueueWriteBuffer(bufferA, CL_FALSE, 0, 1 * sizeof(uint), A.get());
        //queue.enqueueWriteBuffer(bufferB, CL_FALSE, 0, 1 * sizeof(uint), B.get());
        //queue.enqueueWriteBuffer(bufferC, CL_FALSE, 0, 1 * sizeof(uint), C.get());
        queue.enqueueWriteBuffer(bufferD, CL_FALSE, 0, N_ELEMENTS * sizeof(uint), D.get());

        // Read the program source
        std::ifstream sourceFile(KERNEL_FILE);
        std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));

        // Make program from the source code
        cl::Program program = cl::Program(context, source);

        // Build the program for the devices
        program.build(devices);
       
        // Make kernel
        cl::Kernel vecadd_kernel(program, FUNCTION_NAME);
        
        // Set the kernel arguments
        vecadd_kernel.setArg(0, bufferA); // Data
        vecadd_kernel.setArg(1, N_ELEMENTS); // lenght
        vecadd_kernel.setArg(2, 1); // 
        vecadd_kernel.setArg(3, bufferD); // Result
        
        // Execute the kernel
        cl::NDRange global(N_ELEMENTS);
        cl::NDRange local(256);
        
        queue.enqueueNDRangeKernel(vecadd_kernel, cl::NullRange, global, cl::NullRange);
        
        // Copy the output data back to the host
        queue.enqueueReadBuffer(bufferD, CL_TRUE, 0, 1 * sizeof(uint), D.get());

        std::cout << std::hex << D << std::endl;
    }
    catch (cl::Error err) {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        return (EXIT_FAILURE);
    }

    std::cout << "Done.\n";
    return (EXIT_SUCCESS);
}*/