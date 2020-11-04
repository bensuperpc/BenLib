#!/bin/bash
#
# clean.sh - buid in docker ben libs
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
time find . -regex '.*\.\(cpp\|hpp\|c\|h\)' | parallel clang-format -style=file -i {} \;
#time find . -iname *.hpp -o -iname *.h -o -iname *.c | xargs clang-format -style=file -i

#clang-format --verbose -i -style=file *.c
