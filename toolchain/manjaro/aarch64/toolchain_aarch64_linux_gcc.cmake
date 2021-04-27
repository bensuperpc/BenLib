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
set(TOOLCHAIN "aarch64-linux-gnu")

set(CMAKE_ASM_COMPILER "/usr/bin/${TOOLCHAIN}-gcc")

set(CMAKE_C_COMPILER "/usr/bin/${TOOLCHAIN}-gcc")
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS OFF)
set(C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_COMPILER "/usr/bin/${TOOLCHAIN}-g++")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CXX_STANDARD_REQUIRED ON)

# If you change these flags, CMake will not rebuild with these flags
set(CMAKE_ASM_FLAGS_INIT " ${CMAKE_ASM_FLAGS_INIT} -march=armv8-a -mtune=cortex-a72")
set(CMAKE_C_FLAGS_INIT " ${CMAKE_C_FLAGS_INIT} -march=armv8-a -mtune=cortex-a72")
set(CMAKE_CXX_FLAGS_INIT " ${CMAKE_CXX_FLAGS_INIT} -march=armv8-a -mtune=cortex-a72")

#set(CMAKE_C_FLAGS "")
#set(CMAKE_CXX_FLAGS "")

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "C flags")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "C++ flags")


#set(CMAKE_SYSROOT "/usr/aarch64-linux-gnu")

set(CMAKE_FIND_ROOT_PATH "/usr/aarch64-linux-gnu")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
