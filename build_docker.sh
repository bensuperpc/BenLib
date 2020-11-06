#!/bin/bash
#
# build_docker.sh - buid in docker ben libs
#
# Created by Bensuperpc at 31, October of 2020
# Modified by Bensuperpc at 5, November of 2020
#
# Released into the Public domain with MIT licence
# https://opensource.org/licenses/MIT
#
# Written with VisualStudio code 1.49.1
# Script compatibility : Linux and Windows
#
# ==============================================================================

docker build -t ben_lib_builder_manjaro_stable -f config_docker/Dockerfile_manjaro . --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg IMAGE=manjarolinux/base

docker run --rm -it --name ben_lib_builder_00 \
--mount type=bind,source="$(pwd)",destination=/usr/src/app,readonly \
--mount type=bind,source="$(pwd)/build_docker/manjaro/stable",destination=/usr/src/app/build \
ben_lib_builder_manjaro_stable


docker build -t ben_lib_builder_manjaro_unstable -f config_docker/Dockerfile_manjaro . --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg IMAGE=manjarolinux/build-unstable

docker run --rm -it --name ben_lib_builder_01 \
--mount type=bind,source="$(pwd)",destination=/usr/src/app,readonly \
--mount type=bind,source="$(pwd)/build_docker/manjaro/unstable",destination=/usr/src/app/build \
ben_lib_builder_manjaro_unstable



docker build -t ben_lib_builder_archlinux -f config_docker/Dockerfile_archlinux . --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg IMAGE=archlinux

docker run --rm -it --name ben_lib_builder_02 \
--mount type=bind,source="$(pwd)",destination=/usr/src/app,readonly \
--mount type=bind,source="$(pwd)/build_docker/archlinux",destination=/usr/src/app/build \
ben_lib_builder_archlinux


docker build -t ben_lib_builder_ubuntu_20.04 -f config_docker/Dockerfile_ubuntu . --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg IMAGE=ubuntu:20.04

docker run --rm -it --name ben_lib_builder_03 \
--mount type=bind,source="$(pwd)",destination=/usr/src/app,readonly \
--mount type=bind,source="$(pwd)/build_docker/ubuntu/20.04",destination=/usr/src/app/build \
ben_lib_builder_ubuntu_20.04


docker build -t ben_lib_builder_ubuntu_20.10 -f config_docker/Dockerfile_ubuntu . --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg IMAGE=ubuntu:20.10

docker run --rm -it --name ben_lib_builder_04 \
--mount type=bind,source="$(pwd)",destination=/usr/src/app,readonly \
--mount type=bind,source="$(pwd)/build_docker/ubuntu/20.10",destination=/usr/src/app/build \
ben_lib_builder_ubuntu_20.10


docker build -t ben_lib_builder_buster -f config_docker/Dockerfile_ubuntu . --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg IMAGE=debian:buster

docker run --rm -it --name ben_lib_builder_05 \
--mount type=bind,source="$(pwd)",destination=/usr/src/app,readonly \
--mount type=bind,source="$(pwd)/build_docker/debian/buster",destination=/usr/src/app/build \
ben_lib_builder_buster


docker build -t ben_lib_builder_bullseye -f config_docker/Dockerfile_ubuntu . --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg IMAGE=debian:bullseye

docker run --rm -it --name ben_lib_builder_06 \
--mount type=bind,source="$(pwd)",destination=/usr/src/app,readonly \
--mount type=bind,source="$(pwd)/build_docker/debian/bullseye",destination=/usr/src/app/build \
ben_lib_builder_bullseye

docker build -t ben_lib_builder_fedora_33 -f config_docker/Dockerfile_fedora . --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg IMAGE=fedora:33

docker run --rm -it --name ben_lib_builder_07 \
--mount type=bind,source="$(pwd)",destination=/usr/src/app,readonly \
--mount type=bind,source="$(pwd)/build_docker/fedora/33",destination=/usr/src/app/build \
ben_lib_builder_fedora_33

docker build -t ben_lib_builder_fedora_32 -f config_docker/Dockerfile_fedora . --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg IMAGE=fedora:32

docker run --rm -it --name ben_lib_builder_08 \
--mount type=bind,source="$(pwd)",destination=/usr/src/app,readonly \
--mount type=bind,source="$(pwd)/build_docker/fedora/32",destination=/usr/src/app/build \
ben_lib_builder_fedora_32

docker build -t ben_lib_builder_fedora_31 -f config_docker/Dockerfile_fedora . --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg IMAGE=fedora:31

docker run --rm -it --name ben_lib_builder_09 \
--mount type=bind,source="$(pwd)",destination=/usr/src/app,readonly \
--mount type=bind,source="$(pwd)/build_docker/fedora/31",destination=/usr/src/app/build \
ben_lib_builder_fedora_31

docker build -t ben_lib_builder_fedora_34 -f config_docker/Dockerfile_fedora . --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg IMAGE=fedora:34

docker run --rm -it --name ben_lib_builder_10 \
--mount type=bind,source="$(pwd)",destination=/usr/src/app,readonly \
--mount type=bind,source="$(pwd)/build_docker/fedora/34",destination=/usr/src/app/build \
ben_lib_builder_fedora_34