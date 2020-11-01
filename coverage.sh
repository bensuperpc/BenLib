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

gcovr -r . --exclude build/CMakeFiles/ --exclude src/ --html-details -o coverage/coverage.html
#gcovr -r . --exclude src/test/ --exclude src/game.cpp --exclude src/lib/utils/sfml/ --exclude src/bench/ --exclude build/CMakeFiles/
