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
#define FUNCTION_NAME "findStringInv" // findStringInv_MT
//__kernel void CRC32_1byte_tableless(__global const void *data, ulong length, uint previousCrc32, __global uint *resultCrc32)

int main(int argc, char **argv)
{
    // std::cout << crc32Lookup[0][1] << std::endl;
    const int N_ELEMENTS = 29;
    unsigned int platform_id = 0, device_id = 0;

    try {
        //uint64_t B[1] = {8031810176};
        std::unique_ptr<uint64_t> B(new uint64_t);
        B = std::make_unique<uint64_t>(475255);
        //std::unique_ptr<char[]> C(new char[N_ELEMENTS]);
        //auto C = std::make_unique<std::array<char, N_ELEMENTS>>();
        std::unique_ptr<char[]> C = std::unique_ptr<char[]>(new char[N_ELEMENTS]);
        /*
        for(size_t i = 0; i < N_ELEMENTS; i++)
        {
            C[i] = '.';
        }
        std::cout << N_ELEMENTS << std::endl;
        std::cout << *B.get() << std::endl;*/

        //std::unique_ptr<char> C(new char);
        //C = std::make_unique<char>(0);

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
        cl::Buffer bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(uint64_t));
        cl::Buffer bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, N_ELEMENTS * sizeof(char));

        // Copy the input data to the input buffers using the command queue.
        queue.enqueueWriteBuffer(bufferB, CL_FALSE, 0, sizeof(uint64_t), B.get());
        queue.enqueueWriteBuffer(bufferC, CL_FALSE, 0, N_ELEMENTS * sizeof(char), C.get());

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
        vecadd_kernel.setArg(0, bufferB); // lenght
        vecadd_kernel.setArg(1, bufferC);
        // Execute the kernel
        cl::NDRange global(N_ELEMENTS);
        cl::NDRange local(256);
        
        queue.enqueueNDRangeKernel(vecadd_kernel, cl::NullRange, global, cl::NullRange);
        
        // Copy the output data back to the host
        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, N_ELEMENTS * sizeof(char), C.get());

        std::cout << "Result:";
        for(size_t i = 0; i < N_ELEMENTS; i++)
        {
            std::cout << C[i];
        }
        std::cout << std::endl;
    }
    catch (cl::Error err) {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        return (EXIT_FAILURE);
    }

    std::cout << "Done.\n";
    return (EXIT_SUCCESS);
}