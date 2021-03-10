#!/bin/bash
cd build
#find . ! -name '.gitkeep' -type f -exec rm -f {} +
scan-build -o scanbuildout cmake $@ -G Ninja ..
scan-build -o scanbuildout ninja

