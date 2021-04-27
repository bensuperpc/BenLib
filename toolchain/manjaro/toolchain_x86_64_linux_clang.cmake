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

set(CMAKE_ASM_COMPILER "/usr/bin/clang")
set(CMAKE_ASM_COMPILER_TARGET ${TOOLCHAIN})

set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_C_COMPILER_TARGET ${TOOLCHAIN})
set(CMAKE_C_STANDARD 11)

set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_COMPILER_TARGET ${TOOLCHAIN})
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")
#set(CMAKE_CUDA_COMPILER_TARGET "")
set(CMAKE_CUDA_STANDARD 17)

# If you change these flags, CMake will not rebuild with these flags
set(CMAKE_ASM_FLAGS_INIT " ${CMAKE_ASM_FLAGS_INIT} -march=skylake")
set(CMAKE_C_FLAGS_INIT " ${CMAKE_C_FLAGS_INIT} -march=skylake")
set(CMAKE_CXX_FLAGS_INIT " ${CMAKE_CXX_FLAGS_INIT} -march=skylake")

#set(CMAKE_C_FLAGS "")
#set(CMAKE_CXX_FLAGS "")

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "C flags")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "C++ flags")


set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda;/usr;/usr/local/cuda")
set(CUDA_INCLUDE_DIRS "/opt/cuda/include")

include_directories("${CUDA_INCLUDE_DIRS}")

set(CUDA_TOOLKIT_INCLUDE "/opt/cuda/include")
set(CUDA_CUDART_LIBRARY "/opt/cuda/lib64/libcudart.so")
set(CUDA_cublas_LIBRARY "/opt/cuda/lib64/libcublas.so")
set(CUDA_LIBRARIES "/opt/cuda/lib64/libcudart_static.a;Threads::Threads;dl;/usr/lib/librt.so")
set(CUDA_cudart_static_LIBRARY "/opt/cuda/lib64/libcudart_static.a")
set(CUDA_cudadevrt_LIBRARY "/opt/cuda/lib64/libcudadevrt.a")
set(CUDA_NVCC_EXECUTABLE "/opt/cuda/bin/nvcc")
set(CUDA_SDK_ROOT_DIR "CUDA_SDK_ROOT_DIR-NOTFOUND")
set(CUDA_SDK_SEARCH_PATH "CUDA_SDK_ROOT_DIR-NOTFOUND;/opt/cuda;/usr;/usr/local/cuda/local/NVSDK0.2;/opt/cuda;/usr;/usr/local/cuda/NVSDK0.2;/opt/cuda;/usr;/usr/local/cuda/NV_CUDA_SDK;/home/bensuperpc/NVIDIA_CUDA_SDK;/home/bensuperpc/NVIDIA_CUDA_SDK_MACOSX;/Developer/CUDA")
set(CUDA_cufft_LIBRARY "/opt/cuda/lib64/libcufft.so")
set(CUDA_nppc_LIBRARY "/opt/cuda/lib64/libnppc.so")
set(CUDA_nppial_LIBRARY "/opt/cuda/lib64/libnppial.so")
set(CUDA_nppicc_LIBRARY "/opt/cuda/lib64/libnppicc.so")
set(CUDA_nppicom_LIBRARY "")
set(CUDA_nppidei_LIBRARY "/opt/cuda/lib64/libnppidei.so")
set(CUDA_nppif_LIBRARY "/opt/cuda/lib64/libnppif.so")
set(CUDA_nppig_LIBRARY "/opt/cuda/lib64/libnppig.so")
set(CUDA_nppim_LIBRARY "/opt/cuda/lib64/libnppim.so")
set(CUDA_nppist_LIBRARY "/opt/cuda/lib64/libnppist.so")
set(CUDA_nppisu_LIBRARY "/opt/cuda/lib64/libnppisu.so")
set(CUDA_nppitc_LIBRARY "/opt/cuda/lib64/libnppitc.so")
set(CUDA_npps_LIBRARY "/opt/cuda/lib64/libnpps.so")

#set(CMAKE_FIND_ROOT_PATH "")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
