#!/bin/bash
#
# build_docker.sh - buid in docker ben libs
#
# Created by Bensuperpc at 26, April of 2021
# Modified by Bensuperpc at 26, April of 2021
#
# Released into the Public domain with MIT licence
# https://opensource.org/licenses/MIT
#
# Written with VisualStudio code 1.49.1
# Script compatibility : Linux and Windows
#
# ==============================================================================

docker run --rm -it --name ben_lib_builder_00 \
--mount type=bind,source="$(pwd)",destination=/usr/src/app,readonly \
--mount type=tmpfs,destination=/usr/src/app/build,tmpfs-size=128m \
ben_lib_builder_ubuntu_20.04

docker run --rm -it --name ben_lib_builder_01 \
--mount type=bind,source="$(pwd)",destination=/usr/src/app,readonly \
--mount type=tmpfs,destination=/usr/src/app/build,tmpfs-size=128m \
ben_lib_builder_ubuntu_20.10

docker run --rm -it --name ben_lib_builder_02 \
--mount type=bind,source="$(pwd)",destination=/usr/src/app,readonly \
--mount type=tmpfs,destination=/usr/src/app/build,tmpfs-size=128m \
ben_lib_builder_ubuntu_21.04

docker run --rm -it --name ben_lib_builder_03 \
--mount type=bind,source="$(pwd)",destination=/usr/src/app,readonly \
--mount type=tmpfs,destination=/usr/src/app/build,tmpfs-size=128m \
ben_lib_builder_archlinux
