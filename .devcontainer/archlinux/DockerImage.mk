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
IMAGE_NAME := $(PROJECT_NAME)
IMAGE_PATH := $(AUTHOR)
ifeq ($(IMAGE_PATH),)
	OUTPUT_IMAGE := $(IMAGE_NAME)
else
	OUTPUT_IMAGE := $(IMAGE_PATH)/$(IMAGE_NAME)
endif

# Docker config
DOCKERFILE := Dockerfile
DOCKER_EXEC := docker
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

PROGRESS_OUTPUT := plain

ARCH_LIST := linux/amd64
comma:= ,
PLATFORMS := $(subst $() $(),$(comma),$(ARCH_LIST))

# Git config
GIT_SHA := $(shell git rev-parse HEAD)
GIT_ORIGIN := $(shell git config --get remote.origin.url) 

DATE := $(shell date -u +"%Y%m%d")
UUID := $(shell uuidgen)

USER := $(shell whoami)
UID := $(shell id -u ${USER})
GID := $(shell id -g ${USER})

.PHONY: all test push pull run

all: $(addsuffix .test,$(BASE_IMAGE_TAGS))

build: $(BASE_IMAGE_TAGS)

test: $(addsuffix .test,$(BASE_IMAGE_TAGS))

push: $(addsuffix .push,$(BASE_IMAGE_TAGS))

pull: $(addsuffix .pull,$(BASE_IMAGE_TAGS))

.PHONY: $(BASE_IMAGE_TAGS)
$(BASE_IMAGE_TAGS): $(Dockerfile)
	$(DOCKER_EXEC) buildx build . --file $(DOCKERFILE) \
		--platform $(PLATFORMS) --progress $(PROGRESS_OUTPUT) \
		--tag $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$@-$(IMAGE_VERSION)-$(DATE)-$(GIT_SHA) \
		--tag $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$@-$(IMAGE_VERSION)-$(DATE) \
		--tag $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$@-$(IMAGE_VERSION) \
		--tag $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$@ \
		--memory $(BUILD_MEMORY) --cpu-shares $(BUILD_CPU_SHARES) --compress \
		--build-arg BUILD_DATE=$(DATE) --build-arg DOCKER_IMAGE=$(BASE_IMAGE_REGISTRY)/$(BASE_IMAGE_NAME):$@ \
		--build-arg IMAGE_VERSION=$(IMAGE_VERSION) --build-arg PROJECT_NAME=$(PROJECT_NAME) \
		--build-arg VCS_REF=$(GIT_SHA) --build-arg VCS_URL=$(GIT_ORIGIN) \
		--build-arg AUTHOR=$(AUTHOR) --build-arg URL=$(WEB_SITE) \
		$(DOCKER_DRIVER)

.SECONDEXPANSION:
$(addsuffix .build,$(BASE_IMAGE_TAGS)): $$(basename $$@)

.SECONDEXPANSION:
$(addsuffix .test,$(BASE_IMAGE_TAGS)): $$(basename $$@)
	$(DOCKER_EXEC) run --rm \
		--security-opt no-new-privileges --read-only --user $(UID):$(GID) \
		--mount type=bind,source=$(shell pwd),target=/work --workdir /work \
		--mount type=tmpfs,target=/tmp,tmpfs-mode=1777,tmpfs-size=$(TMPFS_SIZE) \
		--platform $(PLATFORMS) \
		--cpus $(CPUS) --cpu-shares $(CPU_SHARES) --memory $(MEMORY) --memory-reservation $(MEMORY_RESERVATION) \
		--name $(IMAGE_NAME)-$(BASE_IMAGE_NAME)-$(basename $@)-$(DATE)-$(GIT_SHA)-$(UUID) \
		$(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$(basename $@)-$(IMAGE_VERSION)-$(DATE)-$(GIT_SHA) \
		$(TEST_CMD)

#--cap-drop ALL --cap-add SYS_PTRACE	  --device=/dev/kvm

.SECONDEXPANSION:
$(addsuffix .run,$(BASE_IMAGE_TAGS)): $$(basename $$@)
	$(DOCKER_EXEC) run -it \
		--security-opt no-new-privileges --read-only --user $(UID):$(GID) \
		--mount type=bind,source=$(shell pwd),target=/work --workdir /work \
		--mount type=tmpfs,target=/tmp,tmpfs-mode=1777,tmpfs-size=$(TMPFS_SIZE) \
		--platform $(PLATFORMS) \
		--cpus $(CPUS) --cpu-shares $(CPU_SHARES) --memory $(MEMORY) --memory-reservation $(MEMORY_RESERVATION) \
		--name $(IMAGE_NAME)-$(BASE_IMAGE_NAME)-$(basename $@)-$(DATE)-$(GIT_SHA)-$(UUID) \
		$(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$(basename $@)-$(IMAGE_VERSION)-$(DATE)-$(GIT_SHA)

.SECONDEXPANSION:
$(addsuffix .push,$(BASE_IMAGE_TAGS)): $$(basename $$@)
	@echo "Pushing $(REGISTRY)/$(OUTPUT_IMAGE)"
	$(DOCKER_EXEC) push $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$(basename $@)
	$(DOCKER_EXEC) push $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$(basename $@)-$(IMAGE_VERSION)
	$(DOCKER_EXEC) push $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$(basename $@)-$(IMAGE_VERSION)-$(DATE)
	$(DOCKER_EXEC) push $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$(basename $@)-$(IMAGE_VERSION)-$(DATE)-$(GIT_SHA)
#   $(DOCKER_EXEC) push $(REGISTRY)/$(OUTPUT_IMAGE) --all-tags

.SECONDEXPANSION:
$(addsuffix .pull,$(BASE_IMAGE_TAGS)): $$(basename $$@)
	@echo "Pulling $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$(basename $@)" 
	$(DOCKER_EXEC) pull $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$(basename $@)
	$(DOCKER_EXEC) pull $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$(basename $@)-$(IMAGE_VERSION)
	$(DOCKER_EXEC) pull $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$(basename $@)-$(IMAGE_VERSION)-$(DATE)
	$(DOCKER_EXEC) pull $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$(basename $@)-$(IMAGE_VERSION)-$(DATE)-$(GIT_SHA)

.PHONY: clean
clean:
	@echo "Clean all untagged images"
	docker system prune -f
#	$(DOCKE) builder prune -f

.PHONY: purge
purge: clean
	@echo "Remove all $(OUTPUT_IMAGE) images and tags"
	$(DOCKER_EXEC) images --filter='reference=$(OUTPUT_IMAGE)' --format='{{.Repository}}:{{.Tag}}' | xargs -r $(DOCKER_EXEC) rmi -f
#   	docker rmi -f $(docker images -f "dangling=true" -q) 2>/dev/null || true

.PHONY: update
update:
#   Update all docker image
	$(foreach tag,$(BASE_IMAGE_TAGS),$(DOCKER_EXEC) pull $(BASE_IMAGE_NAME):$(tag);)
#   Update all submodules to latest
	git submodule update --init --recursive --remote

# https://github.com/linuxkit/linuxkit/tree/master/pkg/binfmt
.PHONY: qemu
qemu:
	export DOCKER_CLI_EXPERIMENTAL=enabled
	$(DOCKER_EXEC) run --rm --privileged multiarch/qemu-user-static --reset -p yes
	$(DOCKER_EXEC) buildx create --name qemu_builder --driver docker-container --use
	$(DOCKER_EXEC) buildx inspect --bootstrap

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all: Build and test all images"
	@echo "  build: Build all images"
	@echo "  test: Test all images"
	@echo "  push: Push all images"
	@echo "  pull: Pull all images"
	@echo "  clean: Clean all untagged images"
	@echo "  purge: Remove all images and tags"
	@echo "  update: Update all images and submodules"
	@echo "  qemu: Install qemu"
	@echo "  help: Show this help message"
	@echo ""
	@echo "  All images: $(BASE_IMAGE_TAGS)"
	@echo "  Sub targets: $(addsuffix .build,$(BASE_IMAGE_TAGS)) $(addsuffix .test,$(BASE_IMAGE_TAGS)) \
	$(addsuffix .push,$(BASE_IMAGE_TAGS)) $(addsuffix .pull,$(BASE_IMAGE_TAGS))"

.SECONDEXPANSION:
$(addsuffix .save,$(BASE_IMAGE_TAGS)): $$(basename $$@)
	@echo "Not implemented yet"
#	docker save $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$(basename $@)-$(IMAGE_VERSION) | xz -e7 -v -T0 > $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$(basename $@)-$(IMAGE_VERSION).tar.xz

#   Bash version
#	DOCKER_IMAGE=ben/ben:ben; install -Dv /dev/null "$DOCKER_IMAGE".tar.xz && docker pull "$DOCKER_IMAGE" && docker save "$DOCKER_IMAGE" | xz -e7 -v -T0 > "$DOCKER_IMAGE".tar.xz

.SECONDEXPANSION:
$(addsuffix .load,$(BASE_IMAGE_TAGS)): $$(basename $$@)
	@echo "Not implemented yet"
#	xz -v -d -k < $(REGISTRY)/$(OUTPUT_IMAGE):$(BASE_IMAGE_NAME)-$(basename $@)-$(IMAGE_VERSION).tar.xz | docker load