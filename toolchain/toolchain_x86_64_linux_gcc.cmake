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
#                                                            #
#  OS: ALL                                                   #
#  CPU: ALL                                                  #
#                                                            #
##############################################################

cmake_minimum_required(VERSION 3.10)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(TOOLCHAIN "x86_64-linux-gnu")

set(CMAKE_ASM_COMPILER gcc)
set(CMAKE_ASM_COMPILER_TARGET ${TOOLCHAIN})

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_C_COMPILER_TARGET ${TOOLCHAIN})

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_COMPILER_TARGET ${TOOLCHAIN})


set(CMAKE_ASM_FLAGS " ${CMAKE_ASM_FLAGS} -march=skylake")
set(CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} -march=skylake")
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -march=skylake")


#set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES )
#set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES )


set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
