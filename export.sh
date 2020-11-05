#!/bin/bash
#
# build_docker.sh - Export docker
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

docker save ben_lib_builder_manjaro_unstable | 7z a -si -m0=lzma2 -mx=9 -mmt12 -ms=on -aoa ben_lib_builder_manjaro_unstable.tar.7z
docker save ben_lib_builder_manjaro_stable | 7z a -si -m0=lzma2 -mx=9 -mmt12 -ms=on -aoa ben_lib_builder_manjaro_stable.tar.7z
docker save ben_lib_builder_archlinux | 7z a -si -m0=lzma2 -mx=9 -mmt12 -ms=on -aoa ben_lib_builder_archlinux.tar.7z 
docker save ben_lib_builder_buster | 7z a -si -m0=lzma2 -mx=9 -mmt12 -ms=on -aoa ben_lib_builder_buster.tar.7z
docker save ben_lib_builder_bullseye | 7z a -si -m0=lzma2 -mx=9 -mmt12 -ms=on -aoa ben_lib_builder_bullseye.tar.7z 
docker save ben_lib_builder_ubuntu_20.04 | 7z a -si -m0=lzma2 -mx=9 -mmt12 -ms=on -aoa ben_lib_builder_ubuntu_20.04.tar.7z 
docker save ben_lib_builder_ubuntu_20.10 | 7z a -si -m0=lzma2 -mx=9 -mmt12 -ms=on -aoa ben_lib_builder_ubuntu_20.10.tar.7z 
docker save ben_lib_builder_fedora_33 | 7z a -si -m0=lzma2 -mx=9 -mmt12 -ms=on -aoa ben_lib_builder_fedora_33.tar.7z 
docker save ben_lib_builder_fedora_32 | 7z a -si -m0=lzma2 -mx=9 -mmt12 -ms=on -aoa ben_lib_builder_fedora_32.tar.7z 
docker save ben_lib_builder_fedora_31 | 7z a -si -m0=lzma2 -mx=9 -mmt12 -ms=on -aoa ben_lib_builder_fedora_31.tar.7z 