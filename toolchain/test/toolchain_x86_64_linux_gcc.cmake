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

set(CMAKE_ASM_COMPILER "gcc")
set(CMAKE_ASM_COMPILER_TARGET ${TOOLCHAIN})

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_C_COMPILER_TARGET ${TOOLCHAIN})

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_COMPILER_TARGET ${TOOLCHAIN})

# If you change these flags, CMake will not rebuild with these flags
set(CMAKE_ASM_FLAGS_INIT " ${CMAKE_ASM_FLAGS_INIT} -march=skylake")
set(CMAKE_C_FLAGS_INIT " ${CMAKE_C_FLAGS_INIT} -march=skylake")
set(CMAKE_CXX_FLAGS_INIT " ${CMAKE_CXX_FLAGS_INIT} -march=skylake")

#set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES )
#set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES )

#CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda;/usr;/opt/cuda")

set(CUDA_TARGET_CPU_ARCH ${CMAKE_SYSTEM_PROCESSOR})

set(CUDA_TARGET_OS_VARIANT "linux")
set(cuda_target_full_path ${CUDA_TARGET_CPU_ARCH}-${CUDA_TARGET_OS_VARIANT})
#set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_TOOLKIT_ROOT_DIR})
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/include)
set(CUDA_TOOLKIT_INCLUDE ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/include)

set(CUDA_CUDART_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/libcudart.so)
set(CUDA_cublas_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libcublas.so)
set(CUDA_cufft_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libcufft.so)
set(CUDA_nppc_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppc.so)
set(CUDA_nppial_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppial.so)
set(CUDA_nppicc_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppicc.so)
set(CUDA_nppicom_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppicom.so)
set(CUDA_nppidei_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppidei.so)
set(CUDA_nppif_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppif.so)
set(CUDA_nppig_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppig.so)
set(CUDA_nppim_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppim.so)
set(CUDA_nppist_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppist.so)
set(CUDA_nppisu_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppisu.so)
set(CUDA_nppitc_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppitc.so)
set(CUDA_npps_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnpps.so)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
