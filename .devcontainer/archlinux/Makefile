#//////////////////////////////////////////////////////////////
#//                                                          //
#//  docker-benlib, 2023                                     //
#//  Created: 04 February, 2023                              //
#//  Modified: 16 June, 2024                                 //
#//  file: -                                                 //
#//  -                                                       //
#//  Source:                                                 //
#//  OS: ALL                                                 //
#//  CPU: ALL                                                //
#//                                                          //
#//////////////////////////////////////////////////////////////

# Base image
BASE_IMAGE_REGISTRY := docker.io
BASE_IMAGE_NAME := archlinux
BASE_IMAGE_TAGS := base

# Output docker image
PROJECT_NAME := benlib
AUTHOR := bensuperpc
REGISTRY := docker.io
WEB_SITE := bensuperpc.org

IMAGE_VERSION := 1.0.0

# Docker config
DOCKER_DRIVER := --load
# --push

# Max CPU and memory
CPUS := 8.0
CPU_SHARES := 1024
MEMORY := 16GB
MEMORY_RESERVATION := 2GB
TMPFS_SIZE := 4GB
BUILD_CPU_SHARES := 1024
BUILD_MEMORY := 16GB

TEST_CMD := ls

ARCH_LIST := linux/amd64


include DockerImage.mk