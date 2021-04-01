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
//          https://sciencing.com/determine-unknown-exponent-8348632.html                                                 //
//          http://www.cplusplus.com/reference/cmath/log/                                                 //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

/**
 *  @addtogroup GTA_SA_GPU_OPENCL
 *  GTA SA Alternate cheat
 *  @brief GTA SA Alternate cheat with OpenCL 1.2
 *  @author Bensuperpc
 * @{
 */
#include <fstream>
#include <iostream>
#include <math.h> // ceil and log
#include <memory> // unique_ptr make_unique
#include <stdlib.h>
#include <string>
#include "string_lib/string_lib_impl.hpp"

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

#define KERNEL_FILE "../kernels/gta_sa.cl"
#define FUNCTION_NAME "findStringInv_T" // findStringInv_MT

int main(int argc, char **argv)
{
    const uint64_t NBRS = 2000000;

    // https://sciencing.com/determine-unknown-exponent-8348632.html
    uint64_t N_ELEMENTS = 0;

    const unsigned int platform_id = 0, device_id = 0;

    cl::Program program;
    std::vector<cl::Device> devices;

    try {

        if constexpr (NBRS > 26) { // Need to be more than 26
            N_ELEMENTS = (uint64_t)ceil(log(NBRS) / log(26));
        } else {
            N_ELEMENTS = 1;
        }
        if constexpr (NBRS > 26) { // Need to be more than 26
            N_ELEMENTS = (uint64_t)ceil(log(NBRS) / log(26));
        } else {
            N_ELEMENTS = 1;
        }

        std::unique_ptr<uint64_t> B(new uint64_t);
        B = std::make_unique<uint64_t>(NBRS);

        std::unique_ptr<char[]> C = std::unique_ptr<char[]>(new char[N_ELEMENTS]);

        std::cout << "N_ELEMENTS: " << N_ELEMENTS << std::endl;
        std::cout << "Nomber: " << *B.get() << std::endl;

        // std::unique_ptr<char> C(new char);
        // C = std::make_unique<char>(0);

        // Query for platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // Get a list of devices on this platform
        // std::vector<cl::Device> devices;
        platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices); // Select the platform.

        // Create a context
        cl::Context context(devices);

        // Create a command queue
        cl::CommandQueue queue = cl::CommandQueue(context, devices[device_id]); // Select the device.

        // Create the memory buffers
        cl::Buffer bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(uint64_t));
        cl::Buffer bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, N_ELEMENTS * sizeof(char));

        // Copy the input data to the input buffers using the command queue.
        queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(uint64_t), B.get());
        queue.enqueueWriteBuffer(bufferC, CL_TRUE, 0, N_ELEMENTS * sizeof(char), C.get());

        // Read the program source
        std::ifstream sourceFile(KERNEL_FILE);
        std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));

        // Make program from the source code
        // cl::Program program;
        program = cl::Program(context, source);

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

        std::cout << "Result GPU: ";
        for (size_t i = 0; i < N_ELEMENTS; i++) {
            std::cout << C[i];
        }
        std::cout << std::endl;

        std::unique_ptr<char[]> res = std::unique_ptr<char[]>(new char[N_ELEMENTS + 1]);
        // char * res = new char[N_ELEMENTS + 1];
        my::string::findStringInv(NBRS, res.get());

        std::cout << "Result CPU: ";
        for (size_t i = 0; i < N_ELEMENTS; i++) {
            std::cout << res[i];
        }
        std::cout << std::endl;
    }
    catch (cl::Error err) {
        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
            // Check the build status
            cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[device_id]);

            // Get the build log
            std::string name = devices[device_id].getInfo<CL_DEVICE_NAME>();
            std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[device_id]);
            std::cerr << "Build log for " << name << ":" << std::endl << buildlog << std::endl;
        }
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        return (EXIT_FAILURE);
    }

    std::cout << "Done.\n";
    return (EXIT_SUCCESS);
}
/** @} */ // end of group2