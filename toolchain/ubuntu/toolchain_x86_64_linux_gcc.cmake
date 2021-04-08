##############################################################
#   ____                                                     #
#  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___    #
#  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __|   #
#  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__    #
#  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___|   #
#                             |_|             |_|            #
##############################################################
#                                                            #
#  BenLib, 2021                                              #
#  Created: 06, April, 2021                                  #
#  Modified: 07, April, 2021                                 #
#  file: CMakeLists.txt                                      #
#  CMake                                                     #
#  Source:      https://stackoverflow.com/questions/15036909/clang-how-to-list-supported-target-architectures                                                   #
#               https://cmake.org/pipermail/cmake/2012-January/048429.html
#               https://stackoverflow.com/questions/11423313/cmake-cross-compiling-c-flags-from-toolchain-file-ignored
#               https://gist.github.com/chaorunrun/06ea22b51e5205bc41a0501d135d053f
#                                                            #
#  OS: ALL                                                   #
#  CPU: ALL                                                  #
#                                                            #
##############################################################


cmake_minimum_required(VERSION 3.10)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(TOOLCHAIN "x86_64-linux-gnu")

set(CMAKE_ASM_COMPILER "/usr/bin/gcc")
set(CMAKE_ASM_COMPILER_TARGET ${TOOLCHAIN})

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_C_COMPILER_TARGET ${TOOLCHAIN})
set(CMAKE_C_STANDARD 11)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_COMPILER_TARGET ${TOOLCHAIN})
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc")
#set(CMAKE_CUDA_COMPILER_TARGET "")
set(CMAKE_CUDA_STANDARD 14)

# If you change these flags, CMake will not rebuild with these flags
set(CMAKE_ASM_FLAGS_INIT " ${CMAKE_ASM_FLAGS_INIT} -march=skylake")
set(CMAKE_C_FLAGS_INIT " ${CMAKE_C_FLAGS_INIT} -march=skylake")
set(CMAKE_CXX_FLAGS_INIT " ${CMAKE_CXX_FLAGS_INIT} -march=skylake")

set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda;/usr;/usr/local/cuda")
set(CUDA_INCLUDE_DIRS "/usr/include")
set(CUDA_TOOLKIT_INCLUDE "/usr/include")
set(CUDA_CUDART_LIBRARY "/usr/lib/x86_64-linux-gnu/libcudart.so")
set(CUDA_cublas_LIBRARY "/usr/lib/x86_64-linux-gnu/libcublas.so")
set(CUDA_LIBRARIES "/usr/lib/x86_64-linux-gnu/libcudart_static.a;Threads::Threads;dl;/usr/lib/x86_64-linux-gnu/librt.so")
set(CUDA_cudart_static_LIBRARY "/usr/lib/x86_64-linux-gnu/libcudart_static.a")
set(CUDA_cudadevrt_LIBRARY "/usr/lib/x86_64-linux-gnu/libcudadevrt.a")
set(CUDA_NVCC_EXECUTABLE "/usr/bin/nvcc")
set(CUDA_SDK_ROOT_DIR "CUDA_SDK_ROOT_DIR-NOTFOUND")
set(CUDA_SDK_SEARCH_PATH "CUDA_SDK_ROOT_DIR-NOTFOUND;/opt/cuda;/usr;/usr/local/cuda/local/NVSDK0.2;/opt/cuda;/usr;/usr/local/cuda/NVSDK0.2;/opt/cuda;/usr;/usr/local/cuda/NV_CUDA_SDK;/home/runner/NVIDIA_CUDA_SDK;/home/runner/NVIDIA_CUDA_SDK_MACOSX;/Developer/CUDA")
set(CUDA_cufft_LIBRARY "/usr/lib/x86_64-linux-gnu/libcufft.so")
set(CUDA_nppc_LIBRARY "/usr/lib/x86_64-linux-gnu/libnppc.so")
set(CUDA_nppial_LIBRARY "/usr/lib/x86_64-linux-gnu/libnppial.so")
set(CUDA_nppicc_LIBRARY "/usr/lib/x86_64-linux-gnu/libnppicc.so")
set(CUDA_nppicom_LIBRARY "/usr/lib/x86_64-linux-gnu/libnppicom.so")
set(CUDA_nppidei_LIBRARY "/usr/lib/x86_64-linux-gnu/libnppidei.so")
set(CUDA_nppif_LIBRARY "/usr/lib/x86_64-linux-gnu/libnppif.so")
set(CUDA_nppig_LIBRARY "/usr/lib/x86_64-linux-gnu/libnppig.so")
set(CUDA_nppim_LIBRARY "/usr/lib/x86_64-linux-gnu/libnppim.so")
set(CUDA_nppist_LIBRARY "/usr/lib/x86_64-linux-gnu/libnppist.so")
set(CUDA_nppisu_LIBRARY "/usr/lib/x86_64-linux-gnu/libnppisu.so")
set(CUDA_nppitc_LIBRARY "/usr/lib/x86_64-linux-gnu/libnppitc.so")
set(CUDA_npps_LIBRARY "/usr/lib/x86_64-linux-gnu/libnpps.so")


set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

