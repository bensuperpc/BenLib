#!/bin/bash
#
# make.sh - Make ben libs
#
# Created by Bensuperpc at 6, October of 2020
#
# Released into the Public domain with MIT licence
# https://opensource.org/licenses/MIT
#
# Written with VisualStudio code 1.49.1
# Script compatibility : Linux and Windows
#
# ==============================================================================

cd build
#cmake .. && make -j 12
#Release/Debug/Coverage/MinSizeRel
#-DCMAKE_BUILD_TYPE=Release
#-DENABLE_CODE_ANALYSIS=O
cmake $@ -G Ninja ..

ninja
#make -j 12
ctest --output-on-failure -j$(nproc) #--extra-verbose

#make install 
