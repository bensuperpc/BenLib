# BenLib

### _It's my personal C/C++ library_

[![N|Solid](https://forthebadge.com/images/badges/made-with-c-plus-plus.svg)](https://isocpp.org/) [![N|Solid](https://forthebadge.com/images/badges/made-with-c.svg)](https://isocpp.org/) [![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com) [![N|Solid](https://forthebadge.com/images/badges/powered-by-qt.svg)](https://www.qt.io/) 

# New Features !

  - AES and RSA encryption functions
  - New linker (Gold linker Instead default linker)
  - GTA SA alternate cheats codes finder, via Brute 
  force: JAMCRC collision generator (WIP)
  - New Readme

### Tech

BenLib uses a number of open source projects to work properly:

* [SFML] - Graphic lib
* [OpenCV] - Load and image processing
* [Boost] - Make units tests and others things :)
* [OpenGL] - OpenGL lib.
* [Qt] - Qt lib.
* [CMake] - Build system.
* [Docker] - Container system.

You can see my [public repository][ben_github] on GitHub, and can see my [public repository][ben_gitlab] on GitLab.

#### Building for source
You need to install SFML, OpenCV, BoostLib, OpenGL lib, Qt 5.12 or newer, GCC and G++ (With C++17 support), before build.

```sh
git clone https://github.com/Bensuperpc/BenLib.git
```

```sh
cd BenLib
```

For production release:
```sh
./make.sh -DCMAKE_BUILD_TYPE=Release
```

For minisize release:
```sh
./make.sh -DCMAKE_BUILD_TYPE=MinSizeRel
```

For debug release:
```sh
./make.sh -DCMAKE_BUILD_TYPE=Release
```

For converage release:
```sh
./make.sh -DCMAKE_BUILD_TYPE=Coverage
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
   [CMake]: <https://cmake.org/>
   [Docker]: <https://www.docker.com/>
   
   [ben_github]: <https://github.com/Bensuperpc>
   [ben_gitlab]: <https://gitlab.com/Bensuperpc>
   [MIT]: LICENSE
   
 
