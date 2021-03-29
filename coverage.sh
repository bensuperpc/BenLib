#!/bin/bash
#
# converage.sh - buid in docker ben libs
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
cmake $@ -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache \
-DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_BUILD_TYPE=Coverage -G Ninja ..

ninja
#make -j 12
ctest --output-on-failure -j$(nproc) #-T Coverage #--extra-verbose
#llvm-cov show bin -instr-profile=code.profdata /path/to/source_files/*.cpp -filename-equivalence -use-color
#cd ..
#gcovr -r . --exclude build/CMakeFiles/ --exclude src/ --html-details -o coverage/coverage.html
#gcovr -r . --exclude src/test/ --exclude src/game.cpp --exclude src/lib/utils/sfml/ --exclude src/bench/ --exclude build/CMakeFiles/
