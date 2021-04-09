#!/bin/bash
#
# make.sh - Make ben libs
#
# Created by Bensuperpc at 6, October of 2020
# Modified by Bensuperpc at 21, March of 2021
#
# Released into the Public domain with MIT licence
# https://opensource.org/licenses/MIT
#
# Written with VisualStudio code 1.49.1
# Script compatibility : Linux and Windows
#
# ==============================================================================

#https://developers.redhat.com/blog/2019/05/15/2-tips-to-make-your-c-projects-compile-3-times-faster/

mkdir -p build
cd build
#cmake .. && make -j$(nproc)
#Release/Debug/Coverage/MinSizeRel
#-DCMAKE_BUILD_TYPE=Release
#-DENABLE_CODE_ANALYSIS=O
#--build build
#cmake build --trace-source="CMakeLists.txt"

cmake -G Ninja $@ .. -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_TOOLCHAIN_FILE=../toolchain/manjaro/toolchain_aarch64_linux_gcc.cmake
#-d explain 
ninja
#ctest --output-on-failure -j$(nproc) #--extra-verbose
#valgrind --tool=callgrind --collect-systime=msec --trace-children=no
#cmake .. --graphviz=foo.dot
#dot -Tpng -Gdpi=120 foo.dot -o foo.png
#make install 
