# BenLib

### _It's my personal C/C++ library_

[![N|Solid](https://forthebadge.com/images/badges/made-with-c-plus-plus.svg)](https://isocpp.org/) [![N|Solid](https://forthebadge.com/images/badges/made-with-c.svg)](https://isocpp.org/) [![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com) [![N|Solid](https://forthebadge.com/images/badges/powered-by-qt.svg)](https://www.qt.io/) 


[![Build test](https://github.com/Bensuperpc/BenLib/actions/workflows/test.yml/badge.svg)](https://github.com/Bensuperpc/BenLib/actions/workflows/test.yml) [![Release Maker](https://github.com/Bensuperpc/BenLib/actions/workflows/release.yml/badge.svg)](https://github.com/Bensuperpc/BenLib/actions/workflows/release.yml) [![GitHub license](https://img.shields.io/github/license/Bensuperpc/BenLib)](https://github.com/Bensuperpc/BenLib/blob/master/LICENSE) [![Activity](https://img.shields.io/github/commit-activity/m/Bensuperpc/BenLib)](https://github.com/Bensuperpc/BenLib/pulse) 

[![Twitter](https://img.shields.io/twitter/follow/Bensuperpc?style=social)](https://img.shields.io/twitter/follow/Bensuperpc?style=social) [![Youtube](https://img.shields.io/youtube/channel/subscribers/UCJsQFFL7QW4LSX9eskq-9Yg?style=social)](https://img.shields.io/youtube/channel/subscribers/UCJsQFFL7QW4LSX9eskq-9Yg?style=social) 

# New Features !

  - AES and RSA encryption functions
  - Add Doxygen doc
  - New linker
  - Add OpenCL and CUDA programs
  - New CRC32 and JAMCRC algo
  - GTA SA alternate cheats codes finder, via Brute 
    force: JAMCRC collisions generator (WIP)
  - CPack with package installer for Debian, Manjaro, Fedora ect...

### Tech

BenLib uses a number of open source projects to work properly:

* [Clang] - Clang 10.0 compiler (or GCC 10 min)
* [CUDA] - Nvidia CUDA libs and NVCC compiler
* [SFML] - Graphic lib
* [OpenCV] - Load and image processing
* [Boost] - Make units tests and others things :)
* [OpenGL] - OpenGL lib.
* [Qt] - Qt lib.
* [CMake] - Build system.
* [OpenMP] - Multi-threading lib. (Not mandatory but really recommended !)
* [OpenCL] - Is a framework for GPGUP
* [Docker] - Container system (if you use it).
* [TLO] - Linker (To replace gold linker)

You can see my [public repository][ben_github] on GitHub, and can see my [public repository][ben_gitlab] on GitLab.

#### Building for source
You need to install SFML, OpenCV, BoostLib, OpenGL lib, Qt 5.12 or newer, GCC and G++ (With C++17 support), before build.

```sh
git clone https://github.com/Bensuperpc/BenLib.git
```
```sh
cd BenLib
```

```sh
git submodule update --init --recursive
```


For production release:
```sh
./make.sh -DCMAKE_BUILD_TYPE=Release -DBUILD_DOCS_DOXYGEN=ON
```

For minisize release:
```sh
./make.sh -DCMAKE_BUILD_TYPE=MinSizeRel
```

For debug release:
```sh
./make.sh -DCMAKE_BUILD_TYPE=Debug -DBUILD_DOCS_DOXYGEN=OFF
```

For converage release:
```sh
./make.sh -DCMAKE_BUILD_TYPE=Coverage -DBUILD_DOCS_DOXYGEN=ON
```
```sh
./coverage.sh
```

### Docker
You must install docker (and docker-compose maybe in later update)

To run docker builder_test :
```sh
./build.sh
```

To export images from docker builder_test (Without builds):

```sh
./export.sh
```

You can see builds on **build_docker/**

### Todos

 - Write MORE Tests
 - Continue dev. :D

License
----

[MIT] License


**Free Software forever !**

   [OpenCV]: <https://opencv.org>
   [SFML]: <https://www.sfml-dev.org>
   [Boost]: <https://www.boost.org>
   [OpenGL]: <https://www.opengl.org>
   [Qt]: <https://www.qt.io/>
   [OpenMP]: <https://www.openmp.org/>
   [CMake]: <https://cmake.org/>
   [OpenCL]: <https://www.khronos.org/opencl/>
   [Docker]: <https://www.docker.com/>
   [TLO]: <https://gcc.gnu.org/wiki/LinkTimeOptimization>
   [Clang]: <https://clang.llvm.org/>
   [CUDA]: <https://developer.nvidia.com/cuda-downloads>
   [ben_github]: <https://github.com/Bensuperpc>
   [ben_gitlab]: <https://gitlab.com/Bensuperpc>
   [MIT]: LICENSE
   
 
