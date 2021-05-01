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
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_CROSSCOMPILING ON)
set(TOOLCHAIN "aarch64-linux-eabi")
set(PLATFORM_ARM "1")


set(CMAKE_ASM_COMPILER "/usr/bin/clang")
set(CMAKE_ASM_COMPILER_TARGET ${TOOLCHAIN})

set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_C_COMPILER_TARGET ${TOOLCHAIN})
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_COMPILER_TARGET ${TOOLCHAIN})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

# If you change these flags, CMake will not rebuild with these flags
set(CMAKE_ASM_FLAGS_INIT " ${CMAKE_ASM_FLAGS_INIT} -march=armv8-a -mtune=cortex-a72")
set(CMAKE_C_FLAGS_INIT " ${CMAKE_C_FLAGS_INIT} -march=armv8-a -mtune=cortex-a72")
set(CMAKE_CXX_FLAGS_INIT " ${CMAKE_CXX_FLAGS_INIT} -march=armv8-a -mtune=cortex-a72")

#set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_C_COMPILER_FORCED ON)
set(CMAKE_CXX_COMPILER_FORCED ON)
set(CMAKE_CUDA_COMPILER_FORCED ON)

#set(CMAKE_LINKER /usr/aarch64-linux-gnu/bin/ld CACHE STRING "Set the cross-compiler tool LD" FORCE)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static --sysroot=${CMAKE_FIND_ROOT_PATH}" CACHE INTERNAL "" FORCE)
set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} --sysroot=${CMAKE_FIND_ROOT_PATH}" CACHE INTERNAL "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static --sysroot=${CMAKE_FIND_ROOT_PATH}" CACHE INTERNAL "" FORCE)
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} --sysroot=${CMAKE_FIND_ROOT_PATH}" CACHE INTERNAL "" FORCE)
#--gcc-toolchain=${GCC_PREFIX}


#if (${CMAKE_VERSION} VERSION_EQUAL "3.6.0" OR ${CMAKE_VERSION} VERSION_GREATER "3.6")
#    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
#else()
#    set(CMAKE_EXE_LINKER_FLAGS_INIT "--specs=nosys.specs")
#endif()


#set(LLVM_TARGETS_TO_BUILD AArch64)

#set(CMAKE_C_FLAGS "")
#set(CMAKE_CXX_FLAGS "")

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "C flags")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "C++ flags")

set(CMAKE_SYSROOT "/usr/aarch64-linux-gnu")
set(CMAKE_FIND_ROOT_PATH "/usr/aarch64-linux-gnu")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
