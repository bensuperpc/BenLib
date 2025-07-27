ARG BASE_IMAGE=debian:stable-slim
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y \
    git \
    gnupg \
    libssl-dev \
    locales \
    curl wget \
    rsync \
    nano \
    build-essential g++ \
    cmake \
    ninja-build \
    gdb \
    clang \
    clang-tidy \
    clang-format \
    valgrind \
    cppcheck \
    libgtest-dev libgmock-dev libbenchmark-dev gcovr \
    bash-completion \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev \
    && apt-get clean autoclean --yes \
    && apt-get autoremove --purge --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}/

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get clean

WORKDIR /workspace
