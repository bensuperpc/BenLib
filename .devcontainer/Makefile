#//////////////////////////////////////////////////////////////
#//                                                          //
#//  docker-benlib, 2023                                         //
#//  Created: 04 February, 2023                              //
#//  Modified: 05 October, 2024                              //
#//  file: -                                                 //
#//  -                                                       //
#//  Source:                                                 //
#//  OS: ALL                                                 //
#//  CPU: ALL                                                //
#//                                                          //
#//////////////////////////////////////////////////////////////

SUBDIRS := debian ubuntu fedora archlinux

# Base image
BASE_IMAGE_REGISTRY := docker.io

# Output docker image
PROJECT_NAME := benlib
AUTHOR := bensuperpc
REGISTRY := docker.io
WEB_SITE := bensuperpc.org

IMAGE_VERSION := 1.0.0
IMAGE_NAME := $(PROJECT_NAME)
ifeq ($(AUTHOR),)
	OUTPUT_IMAGE := $(IMAGE_NAME)
else
	OUTPUT_IMAGE := $(AUTHOR)/$(IMAGE_NAME)
endif

USER := $(shell whoami)
UID := $(shell id -u ${USER})
GID := $(shell id -g ${USER})

# Max CPU and memory
CPUS := 8.0
CPU_SHARES := 1024
MEMORY := 16GB
MEMORY_RESERVATION := 2GB
TMPFS_SIZE := 4GB
BUILD_CPU_SHARES := 1024
BUILD_MEMORY := 16GB

TEST_CMD := ./tests/tests.sh

# linux/amd64,linux/amd64/v3, linux/arm64, linux/riscv64, linux/ppc64
ARCH_LIST := linux/amd64
comma:= ,
PLATFORMS := $(subst $() $(),$(comma),$(ARCH_LIST))

# Docker config
DOCKERFILE := Dockerfile
DOCKER_EXEC := docker
DOCKER_DRIVER := --load
PROGRESS_OUTPUT := plain

# Git config
GIT_SHA := $(shell git rev-parse HEAD)
GIT_ORIGIN := $(shell git config --get remote.origin.url) 

DATE := $(shell date -u +"%Y%m%d")
UUID := $(shell uuidgen)

# Custom targets
CUSTOM_TARGETS := help

# Merge all variables
MAKEFILE_VARS := PROJECT_NAME=$(PROJECT_NAME) AUTHOR=$(AUTHOR) REGISTRY=$(REGISTRY) \
	IMAGE_VERSION=$(IMAGE_VERSION) PLATFORMS="$(PLATFORMS)" \
	CPUS=$(CPUS) CPU_SHARES=$(CPU_SHARES) MEMORY=$(MEMORY) MEMORY_RESERVATION=$(MEMORY_RESERVATION) \
	BUILD_CPU_SHARES=$(BUILD_CPU_SHARES) BUILD_MEMORY=$(BUILD_MEMORY) \
	WEB_SITE=$(WEB_SITE) BASE_IMAGE_REGISTRY=$(BASE_IMAGE_REGISTRY) \
	DOCKERFILE=$(DOCKERFILE) DOCKER_EXEC=$(DOCKER_EXEC) DOCKER_DRIVER=$(DOCKER_DRIVER) \
	GIT_SHA=$(GIT_SHA) GIT_ORIGIN=$(GIT_ORIGIN) DATE=$(DATE) UUID=$(UUID) \
	USER=$(USER) UID=$(UID) GID=$(GID) TMPFS_SIZE=$(TMPFS_SIZE) \
	TEST_CMD=$(TEST_CMD) PROGRESS_OUTPUT=$(PROGRESS_OUTPUT) \
	OUTPUT_IMAGE=$(OUTPUT_IMAGE) IMAGE_NAME=$(IMAGE_NAME)

.PHONY: all clean build test purge update push pull $(SUBDIRS) $(CUSTOM_TARGETS)

default: all

$(SUBDIRS):
	$(MAKE) $(MAKEFILE_VARS) -C $@ all

all: $(addsuffix -all, $(SUBDIRS))

%.all:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.all,%,$@) all

build: $(addsuffix .build, $(SUBDIRS))

%.build:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.build,%,$@) build

test: $(addsuffix .test, $(SUBDIRS))

%.test:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.test,%,$@) test

clean: $(addsuffix .clean, $(SUBDIRS))

%.clean:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.clean,%,$@) clean

purge: $(addsuffix .purge, $(SUBDIRS))

%.purge:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.purge,%,$@) purge

update: $(addsuffix .update, $(SUBDIRS))

%.update:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.update,%,$@) update

push: $(addsuffix .push, $(SUBDIRS))

%.push:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.push,%,$@) push

pull: $(addsuffix .pull, $(SUBDIRS))

%.pull:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.pull,%,$@) pull

$(CUSTOM_TARGETS): $(addsuffix .$(CUSTOM_TARGETS), $(SUBDIRS))

%.$(CUSTOM_TARGETS):
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.$(CUSTOM_TARGETS),%,$@) $(CUSTOM_TARGETS)
