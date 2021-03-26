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
cmake $@ -G Ninja .. -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache \
-DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_BUILD_TYPE=Release

ninja
#ctest --output-on-failure -j$(nproc) #--extra-verbose


#make install 