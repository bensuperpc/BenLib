#!/bin/bash
#
# valgrind.sh - Valgrind ben libs
#
# Created by Bensuperpc at 02, April of 2021
# Modified by Bensuperpc at 02, April of 2021
#
# Released into the Public domain with MIT licence
# https://opensource.org/licenses/MIT
#
# Written with VisualStudio code 1.49.1
# Script compatibility : Linux and Windows
#
# ==============================================================================

valgrind --leak-check=full --track-origins=yes --show-leak-kinds=all -v $1