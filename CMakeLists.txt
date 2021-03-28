##############################################################
#   ____                                                     #
#  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___    #
#  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __|   #
#  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__    #
#  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___|   #
#                             |_|             |_|            #
##############################################################
#                                                            #
#  BenLib, 2020                                              #
#  Created: 19, March, 2020                                  #
#  Modified: 28, March, 2021                                 #
#  file: CMakeLists.txt                                      #
#  CMake                                                     #
#  Source:  https://medium.com/@mikkelwilson/opencv-with-cuda-acceleration-test-e9f7155e1c84                                                   #
#           https://onqtam.com/programming/2019-12-20-pch-unity-cmake-3-16/
#           https://stackoverflow.com/questions/61548932/force-cmake-to-rebuild-precompiled-headers
#           https://www.virag.si/2015/07/use-ccache-with-cmake-for-faster-compilation/
#           https://cmake.org/cmake/help/latest/module/CPack.html#variable:CPACK_BINARY_%3CGENNAME%3E
#           https://cmake.org/cmake/help/v3.0/module/CPack.html
#           https://gitlab.kitware.com/cmake/community/-/wikis/doc/cpack/PackageGenerators
#           https://cmake.org/cmake/help/v3.0/module/CPackRPM.html
#           https://blog.usejournal.com/creating-debian-packages-cmake-e519a0186e87
#           https://stackoverflow.com/questions/49644273/cmake-change-stack-size
#           https://stackoverflow.com/questions/46713143/cmake-install-library-and-development-files-as-separate-components-using-public
#           https://github.com/Pseudomanifold/cmake-cpp
#           https://cmake.org/cmake/help/v3.1/prop_tgt/C_EXTENSIONS.html#prop_tgt:C_EXTENSIONS
#           
#  OS: ALL                                                   #
#  CPU: ALL                                                  #
#                                                            #
##############################################################

cmake_minimum_required (VERSION 3.13 FATAL_ERROR)
#project(BENLIB VERSION 0.0.5 LANGUAGES CUDA CXX C)
project(BENLIB VERSION 0.0.5 DESCRIPTION "Bensuperpc's libs" LANGUAGES CXX C)

#Add CUDA if it found
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
    message(STATUS "CMAKE_CUDA_COMPILER: True")
endif()


#   ==================================================
#   ================    CMAKE CONF    ================
#   ==================================================

option(ENABLE_CCACHE "Whether to use ccache to speed up compiling" ON)
# Include
option(CMAKE_INCLUDE_CURRENT_DIR ON)
# Increase verbose CMake
option(CMAKE_VERBOSE_MAKEFILE ON)
# Have cmake create a compile database
option(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#Disable precompiled header, save space
option(CMAKE_DISABLE_PRECOMPILE_HEADERS ON)


if (CMAKE_BUILD_TYPE)
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()


# If support unity build, improve speed on headers
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set(CMAKE_UNITY_BUILD 1)
    message(STATUS "CMAKE_UNITY_BUILD : 1")
else()
    set(CMAKE_UNITY_BUILD 0)
    message(STATUS "CMAKE_UNITY_BUILD : 0")
endif()

# Add Postfix for debug conf
set(CMAKE_DEBUG_POSTFIX "_d")

#set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
#set(CMAKE_C_COMPILER_LAUNCHER ccache)
#set(Var1 "This is variable 1" CACHE INTERNAL "")
#set(Var1 "This is variable 1" PARENT_SCOPE)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND AND ENABLE_CCACHE)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache) # Less useful to do it for linking, see edit2
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
    set(CMAKE_C_COMPILER_LAUNCHER ccache)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_INSTALL_INCLUDEDIR "include")
#set(PROJECT_BINARY_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
#set(CTEST_BINARY_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#   ================================================
#   ================    COMPILER    ================
#   ================================================
# Enable CXX standard required
option(CMAKE_CXX_STANDARD_REQUIRED ON)
# Enable C standard required
option(CMAKE_C_STANDARD_REQUIRED ON)
option(CMAKE_CXX_EXTENSIONS OFF)
option(CMAKE_C_EXTENSIONS OFF)

option(AUTODETECT_CXX_MAX_VERSION "Detect automatically CXX Max supported version by compiler" ON)
option(AUTODETECT_C_MAX_VERSION "Detect automatically C Max supported version by compiler" ON)
# Shared libs, decrease bin size
option(BUILD_SHARED_LIBRARIES ON)

if(NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER /usr/bin/clang)
#    set(CMAKE_C_COMPILER /usr/bin/gcc)
endif()
if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER /usr/bin/clang++)
#    set(CMAKE_CXX_COMPILER /usr/bin/g++)
endif()

#Set the highest C++ standard supported by the compiler
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_C_STANDARD 11)

#=== C VERSION CHECK ===
# Set the highest C standard supported by the compiler

if(NOT CMAKE_C_STANDARD AND AUTODETECT_C_MAX_VERSION)
    include(CheckCCompilerFlag)
    CHECK_C_COMPILER_FLAG("-std=c23" COMPILER_SUPPORTS_C23)
    CHECK_C_COMPILER_FLAG("-std=c20" COMPILER_SUPPORTS_C20)
    CHECK_C_COMPILER_FLAG("-std=c18" COMPILER_SUPPORTS_C18)
    CHECK_C_COMPILER_FLAG("-std=c17" COMPILER_SUPPORTS_C17) # Not working with CMake 3.19
    CHECK_C_COMPILER_FLAG("-std=c11" COMPILER_SUPPORTS_C11)
    CHECK_C_COMPILER_FLAG("-std=c99" COMPILER_SUPPORTS_C99)
    CHECK_C_COMPILER_FLAG("-std=c90" COMPILER_SUPPORTS_C90)
    CHECK_C_COMPILER_FLAG("-std=c89" COMPILER_SUPPORTS_C89)
    
    if(COMPILER_SUPPORTS_C11)
        message(STATUS "C11: OK")
        set(CMAKE_C_STANDARD 11)
    elseif(COMPILER_SUPPORTS_C99)
        message(STATUS "C99: OK")
        set(CMAKE_C_STANDARD 99)
    elseif(COMPILER_SUPPORTS_C90)
        message(STATUS "C90: OK")
        set(CMAKE_C_STANDARD 90)
    elseif(COMPILER_SUPPORTS_C89)
        message(STATUS "C89: OK")
        set(CMAKE_C_STANDARD 89)
    else()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 or above support. Please use a different C++ compiler.")
    endif()
endif()


#=== C++ VERSION CHECK ===
if(NOT CMAKE_CXX_STANDARD AND AUTODETECT_CXX_MAX_VERSION)
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-std=c++23" COMPILER_SUPPORTS_CXX23)
    CHECK_CXX_COMPILER_FLAG("-std=c++20" COMPILER_SUPPORTS_CXX20)
    CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
    CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
    CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
    CHECK_CXX_COMPILER_FLAG("-std=c++03" COMPILER_SUPPORTS_CXX03)
    CHECK_CXX_COMPILER_FLAG("-std=c++98" COMPILER_SUPPORTS_CXX98)

    if(COMPILER_SUPPORTS_CXX20)
        set(CMAKE_CXX_STANDARD 20)
        message(STATUS "C++20: OK")
    elseif(COMPILER_SUPPORTS_CXX17)
        set(CMAKE_CXX_STANDARD 17)
        message(STATUS "C++17: OK")
    elseif(COMPILER_SUPPORTS_CXX14)
        set(CMAKE_CXX_STANDARD 14)
        message(WARNING "C++14: Error")
        message(FATAL_ERROR "C++14 is old, please use newer compiler.")
    elseif(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_STANDARD 11)
        message(WARNING "C++11: OK")
        message(FATAL_ERROR "C++11 is old, please use newer compiler.")
    elseif(COMPILER_SUPPORTS_CXX03)
        set(CMAKE_CXX_STANDARD 03)
        message(WARNING "C++03: OK")
        message(FATAL_ERROR "C++03 is old, please use newer compiler.")
    elseif(COMPILER_SUPPORTS_CXX98)
        set(CMAKE_CXX_STANDARD 98)
        message(WARNING "C++98: OK")
        message(FATAL_ERROR "C++98 is old, please use newer compiler.")
    else()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 or above support. Please use a different C++ compiler.")
    endif()

    # Compiler-specific C++11 activation.
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        # require at least gcc 4.8
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
            message(FATAL_ERROR "GCC version must be at least 4.8!")
            set(CMAKE_CXX_STANDARD 14)
        endif()
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        # require at least clang 3.2
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.2)
            message(FATAL_ERROR "Clang version must be at least 3.2!")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0")
        message(FATAL_ERROR "Insufficient msvc version")
        endif()
    else()
        message(WARNING "You are using an unsupported compiler! Compilation has only been tested with Clang and GCC.")
    endif()
endif()


#=== SECURITY RELEASE FLAGS ===
set(SEC_COMPILER_REL "-fstack-clash-protection -fstack-protector-all -Werror=format-security -fstack-protector-strong -fexceptions -D_FORTIFY_SOURCE=2 -fPIE")

#=== C FLAGS ===
set(WARNINGS_COMPILER_C "-Wall -Wpedantic -Wextra -Wstrict-prototypes -Wmissing-prototypes -Wfloat-equal -Wundef -Wshadow -Wpointer-arith -Wstrict-overflow=5 -Wswitch-default -Wunreachable-code -Wcast-align")

set(CMAKE_C_FLAGS                " -march=native -ffunction-sections -fdata-sections ${WARNINGS_COMPILER_C} -pipe ")
set(CMAKE_C_FLAGS_RELEASE        "-O3 ${SEC_COMPILER_REL}")
set(CMAKE_C_FLAGS_MINSIZEREL     "-Os ${SEC_COMPILER_REL}")
set(CMAKE_C_FLAGS_DEBUG          "-g3 -Og -rdynamic -ggdb3 -pg") # Remove -v
set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_C_FLAGS_DEBUG} --coverage")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_C_FLAGS                " ${CMAKE_C_FLAGS} -flto=thin")
endif()

#=== CXX FLAGS ===
#-Wold-style-cast -Wdouble-promotion -fstack-usage -Wpadded

#add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wshadow>")


set(WARNINGS_COMPILER_CXX "${WARNINGS_COMPILER_CXX} -Wall -Wextra -Wpedantic -Wshadow -Wmissing-declarations -Wundef -Wstack-protector -Wno-unused-parameter")
set(WARNINGS_COMPILER_CXX "${WARNINGS_COMPILER_CXX} -Wmissing-include-dirs -Wmissing-noreturn -Wimport -Winit-self -Winvalid-pch -Wstrict-aliasing=2 -Wswitch-default -Wunreachable-code -Wunused")
set(WARNINGS_COMPILER_CXX "${WARNINGS_COMPILER_CXX} -Woverloaded-virtual  -Wdisabled-optimization -Winline -Wredundant-decls -Wsign-conversion -Wformat-nonliteral -Wformat-security")
set(WARNINGS_COMPILER_CXX "${WARNINGS_COMPILER_CXX} -Wwrite-strings -Wcast-align -Wcast-qual -Wfloat-equal -Wvariadic-macros -Wpacked -Wpointer-arith -Weffc++ -Wformat=2 -Wfloat-equal")
set(WARNINGS_COMPILER_CXX "${WARNINGS_COMPILER_CXX} -Wnull-dereference")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(WARNINGS_COMPILER_CXX "${WARNINGS_COMPILER_CXX} -Wabi=11 -Wduplicated-branches -Wduplicated-cond -Wlogical-op")
endif()


# Removed : -Wconversion -Wuseless-cast
#-ftime-report -static -lrt -pthread -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -flto

set(CMAKE_CXX_FLAGS                "  -ffunction-sections -fdata-sections -march=native ${WARNINGS_COMPILER_CXX} -pipe")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS                " ${CMAKE_CXX_FLAGS} -lstdc++fs")
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS                " ${CMAKE_CXX_FLAGS} -flto=thin")
endif()

set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os ${SEC_COMPILER_REL} -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 ${SEC_COMPILER_REL} -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG          "-g3 -Og -rdynamic -ggdb3 -pg") # Remove -v
set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} --coverage")


set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld -s -fPIC -Wl,-z,now -Wl,-z,relro -Wl,--sort-common,--as-needed,--gc-sections,--strip-all,--allow-multiple-definition -Wl,-rpath,../lib -Wl,-rpath,../external/lib -Wl,-rpath,../../lib ")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_EXE_LINKER_FLAGS                " ${CMAKE_EXE_LINKER_FLAGS} -flto=thin -Wl,--thinlto-jobs=all")
endif()

if (CMAKE_GENERATOR STREQUAL "Ninja")
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fcolor-diagnostics")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-color=always")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
  endif()
endif()

add_compile_options(
    "$<$<CXX_COMPILER_ID:Clang>:>"
    "$<$<CXX_COMPILER_ID:Gnu>:>"
    "$<$<CXX_COMPILER_ID:MSVC>:>"
    )
#add_compile_options(
#    $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CXX_FLAGS}>
#    $<$<COMPILE_LANGUAGE:C>:${c_flags}>
#    )
#target_compile_options(my-target
#  PRIVATE
#    $<$<CXX_COMPILER_ID:Gnu>:
#      # g++ warning flags
#    >
#    $<$<CXX_COMPILER_ID:Clang>:
#      # clang warning flags
#    >
#    $<$<CXX_COMPILER_ID:MSVC>:
#      # MSVC warning flags
#    >
#)


#   ================    COMPILER DEF    ================

add_definitions(-DPROJECT_VERSION="${PROJECT_VERSION}")
add_definitions(-DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
add_definitions(-DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}")
add_definitions(-DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}")
add_definitions(-DCMAKE_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}")
add_definitions(-DCMAKE_C_COMPILER_ID="${CMAKE_C_COMPILER_ID}")
add_definitions(-DCMAKE_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}")
add_definitions(-DCMAKE_C_COMPILER_VERSION="${CMAKE_C_COMPILER_VERSION}")
add_definitions(-DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS}")


if(CMAKE_SIZEOF_VOID_P EQUAL 2)
    message(STATUS "16 bit mode")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(STATUS "32 bit mode")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "64 bit mode")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 16)
    message(STATUS "128 bit mode")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 32)
    message(STATUS "256 bit mode")
else()
    message(STATUS "unknow size")
endif()

#   =================================================
#   ==============    EXTERNAL LIBS    ==============
#   =================================================

include(ExternalProject)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

#   ================    OpenMP    ================
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

#   ================    OpenMP    ================
# https://stackoverflow.com/a/12404666/10152334
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
# OpenMP::OpenMP_CXX
find_package(OpenMP QUIET)

if (OPENMP_FOUND OR OpenMP_CXX_FOUND)
    message(STATUS "OPENMP: FOUND")
    #set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    #set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    #set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(STATUS "OPENMP: NOT FOUND")
endif()

#   ================    SFML    ================
#set(SFML_USE_STATIC_STD_LIBS 0)
#set(SFML_STATIC_LIBRARIES 0)

set(SFML_ROOT "C:\\Project Files (x86)\\SFML")

find_package(SFML 2.5 COMPONENTS graphics audio network system QUIET)

if(SFML_FOUND)
    message(STATUS "SFML: FOUND")
    add_compile_definitions(SFML_VERSION="${SFML_VERSION}")
else()
    message(STATUS "SFML: NOT FOUND")
    #ExternalProject_Add(sfml_lib
    #    GIT_REPOSITORY https:#github.com/SFML/SFML.git
    #    GIT_TAG 2.5.1
    #    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
        #CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    #    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
    #)
endif()

#   ================    BOOST    ================
if (WIN32)
    set(Boost_USE_STATIC_LIBS TRUE)
    set(Boost_USE_MULTITHREADED TRUE)
    set(Boost_USE_STATIC_RUNTIME FALSE)
else ()
    set(Boost_USE_STATIC_LIBS FALSE)
    set(Boost_USE_MULTITHREADED TRUE)
    set(Boost_USE_STATIC_RUNTIME FALSE)
endif (WIN32)

#find_package( Boost 1.72.0 COMPONENTS thread system fiber context program_options filesystem REQUIRED)
find_package(Boost 1.67.0 COMPONENTS thread filesystem system unit_test_framework QUIET)

if(Boost_FOUND)
    message(STATUS "Boost FOUND")
    include_directories( ${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARIES})
else()
    message(STATUS "Boost: NOT FOUND")
endif()

#   ================    OPENCV    ================
find_package(OpenCV REQUIRED QUIET)

if (OPENCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OPENCV: FOUND")
else()
    message(STATUS "OPENCV: NOT FOUND")
    if (WIN64)
        set(OpenCV_DIR "C:/Users/Benoit/Downloads/opencv-4.4.0/build")
    endif (WIN64)
    #ExternalProject_Add(opencv_lib
    #    GIT_REPOSITORY https://github.com/opencv/opencv
    #    GIT_TAG 4.4.0
    #    SOURCE_DIR opencv
    #    BINARY_DIR opencv-build
    #    CMAKE_ARGS -D CMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    #    CMAKE_ARGS -D BUILD_EXAMPLES=OFF -D OPENCV_ENABLE_NONFREE=ON
    #    CMAKE_ARGS -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF
        #CMAKE_ARGS -DOPENCV_EXTRA_MODULES_PATH=/build/opencv_contrib-4.1.1/modules
        #CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        #-D WITH_CUDA=ON
    #    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
    #)
endif()


#   ================    OPENGL    ================
find_package(OpenGL REQUIRED QUIET)
find_package(GLUT REQUIRED QUIET)

if (OPENGL_FOUND)
    include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
    link_directories(${OPENGL_LIBRARIES})
    message(STATUS "OPENGL: FOUND")
else()
    message(STATUS "OPENGL: NOT FOUND")
endif()

#   ================    OPENCL    ================

find_package(OpenCL REQUIRED QUIET)

if (OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS})
    if (UNIX)
        include_directories(/opt/cuda/targets/x86_64-linux/include/)
    endif()
    #link_directories(/opt/cuda/targets/x86_64-linux/lib/)
    link_directories(${OpenCL_LIBRARIES})
    message(STATUS "OPENCL: FOUND")
    
else()
    message(STATUS "OPENCL: NOT FOUND")
endif()

#   ================    CUDA    ================
# https://bugs.archlinux.org/task/62258
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
#export PATH=/opt/cuda/bin:$PATH
#export LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

#set(CUTLASS_NATIVE_CUDA_INIT ON)
find_package(CUDA 10.0 QUIET)
#set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda")
if (CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARIES})
    #set(CMAKE_CUDA_FLAGS	"${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall")
    #set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")
    #set(CMAKE_CUDA_STANDARD 17)

    message(STATUS "CUDA: FOUND")
    message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
else()
    message(STATUS "CUDA: NOT FOUND")
endif()

#set_target_properties(BENLIB PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


#   ================    OpenACC    ===============
# Unsupported by intel and AMD GPU
#find_package(OpenACC QUIET)

#if (OpenACC_CXX_FOUND)
#    message(STATUS "OpenACC: FOUND")
#else()
#    message(STATUS "OpenACC: NOT FOUND")
#endif()
#${OpenACC_CXX_FLAGS}

#   ================    VULKAN    ================
# Unnecessary for now
find_package(Vulkan QUIET)

if (VULKAN_FOUND)
    include_directories( ${Vulkan_INCLUDE_DIR})
    link_directories(${Vulkan_LIBRARY})
    message(STATUS "VULKAN: FOUND")
else()
    message(STATUS "VULKAN: NOT FOUND")
endif()

#   ================    OPENSSL    ================

find_package(OpenSSL QUIET)

if (OPENSSL_FOUND)
    include_directories( ${OPENSSL_INCLUDE_DIR})
    link_directories(${OPENSSL_LIBRARIES})
    message(STATUS "OPENSSL: FOUND")
else()
    message(STATUS "OPENSSL: NOT FOUND")
endif()

#   ================    Qt    ================

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(QT NAMES Qt6 Qt5 COMPONENTS Core Quick Multimedia)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core Quick Multimedia)
find_package(Qt${QT_VERSION_MAJOR}QuickCompiler)
#qtquick_compiler_add_resources(RESOURCES example.qrc)
#qt5_use_modules(myapp Quick Widgets Core Gui Multimedia Network)

#find_package(Qt5 COMPONENTS Quick Widgets Core Gui Multimedia Network REQUIRED)

if (QT_FOUND)
    message(STATUS "QT: FOUND")
    include_directories( ${QT_INCLUDES})
else()
    message(STATUS "QT: NOT FOUND")
endif()


#   ================    cppcheck    ================

# https://stackoverflow.com/a/63840110/10152334

option(ENABLE_CODE_ANALYSIS "Run code analysis" OFF)
message(STATUS "ENABLE_CODE_ANALYSIS: ${ENABLE_CODE_ANALYSIS}")

if(ENABLE_CODE_ANALYSIS)
    find_program(cppcheck cppcheck)
    message(STATUS "cppcheck                   ${cppcheck}")
    if(NOT (cppcheck MATCHES "NOTFOUND"))
        # Notes:
        # - unmatchedSuppression needed since not all source files are
        #   analyzed simultaneously
        # - unusedFunction needed since not all source files are
        #   analyzed simultaneously
        # - template needed to force "warning" into output string, and
        #   to make the 'id' available for suppression
        set(CMAKE_CXX_CPPCHECK "${cppcheck}"
            "--enable=all"
            "--inconclusive"
            "--inline-suppr"
            "--quiet"
            "--suppress=unmatchedSuppression"
            "--suppress=unusedFunction"
            "--template='{file}:{line}: warning: {id} ({severity}): {message}'")
    endif()
endif(ENABLE_CODE_ANALYSIS)

#   =================================================
#   =============    INTERNAL LIBS    ===============
#   =================================================
#set(BASEPATH "${CMAKE_SOURCE_DIR}")
#include_directories("${BASEPATH}")

set(lib_path "")

include_directories("${lib_path}lib/") #To avoid to use full path in headers
include_directories("${lib_path}lib/C/") #To avoid to use full path in headers


# C libs
add_subdirectory(${lib_path}lib/C/libhello "src/lib/lib/C/libhello")
add_subdirectory(${lib_path}lib/C/string_c "src/lib/lib/C/string_c")

add_subdirectory(${lib_path}lib/C/string_is "src/lib/lib/C/string_is_c")
add_subdirectory(${lib_path}lib/C/pointer "src/lib/lib/C/pointer_c")
add_subdirectory(${lib_path}lib/C/display "src/lib/lib/C/display_c")
add_subdirectory(${lib_path}lib/C/integer "src/lib/lib/C/integer_c")

# C++ libs
add_subdirectory(${lib_path}lib/convert_utils "src/lib/lib/convert_utils")

add_subdirectory(${lib_path}lib/texture "src/lib/lib/texture")

add_subdirectory(${lib_path}lib/math "src/lib/lib/math")

add_subdirectory(${lib_path}lib/utils/opencv "src/lib/lib/utils/opencv")

add_subdirectory(${lib_path}lib/utils/opengl "src/lib/lib/utils/opengl")

#add_subdirectory(src/lib/utils/sfml "src/lib/lib/utils/sfml")

add_subdirectory(${lib_path}lib/time "src/lib/lib/time")

add_subdirectory(${lib_path}lib/cpu "src/lib/lib/cpu")

add_subdirectory(${lib_path}lib/vector "src/lib/lib/vector")

add_subdirectory(${lib_path}lib/thread "src/lib/lib/thread")

add_subdirectory(${lib_path}lib/filesystem "src/lib/lib/filesystem")

add_subdirectory(${lib_path}lib/string_lib "src/lib/lib/string_lib")
if (OPENSSL_FOUND)
    add_subdirectory(${lib_path}lib/crypto "src/lib/lib/crypto")
endif()
add_subdirectory(${lib_path}lib/std "src/lib/lib/std")

if (CUDA_FOUND)
    add_subdirectory(${lib_path}lib/CUDA "src/lib/lib/CUDA")
endif()

#add_dependencies(hello ${EXTERNAL_LIB})



#   =================================================
#   =============    SOURCES FILES    ===============
#   =================================================

#   ================    FILES    ================
set(MAIN src/main.cpp)

set(SRCS
    )

set(HEADERS
    )

set(VIRUS 
    )

set(BENCH_ISPRIME ${lib_path}src/bench/bench_isprime.cpp)

set(BENCH_VECTOR_COPY ${lib_path}src/bench/bench_vector_copy.cpp)

set(BENCH_VECTOR_SORT ${lib_path}src/bench/bench_vector_sort.cpp)

set(BENCH_DIGITS_COUNT ${lib_path}src/bench/bench_digits_count.cpp)

set(BENCH_FIND_MAX ${lib_path}src/bench/bench_find_max.cpp)

set(BENCH_PC ${lib_path}src/bench/benchmark_pc.cpp)

set(BENCH_CRC32 ${lib_path}src/bench/bench_crc32.cpp)

set(LIST_ALL_FILE ${lib_path}src/exemple/list_all_file.cpp)

set(PARSE_CSV ${lib_path}src/exemple/parse_csv.cpp)

set(GTA_SA_CHEAT ${lib_path}src/exemple/GTA_SA_cheat.cpp)

if (OpenCL_FOUND)
    set(OPENCL_TEST ${lib_path}src/exemple/OpenCL/OpenCL_test.cpp)
    set(OPENCL_TEST2 ${lib_path}src/exemple/OpenCL/OpenCL_test2.cpp)
    set(OPENCL_CRC32 ${lib_path}src/exemple/OpenCL/OpenCL_crc32.cpp)
    set(OPENCL_FINDSTRINGINV ${lib_path}src/exemple/OpenCL/OpenCL_findStringInv.cpp)
    set(OPENCL_GTA_SA_GPU ${lib_path}src/exemple/OpenCL/OpenCL_GTA_SA_cheat_GPU.cpp)
endif()

if (OPENSSL_FOUND)
    set(LIST_ALL_FILE_HASH ${lib_path}src/exemple/list_all_file_hash.cpp)
    set(CRYPTO_AES ${lib_path}src/exemple/crypto_AES256.cpp)
    set(CRYPTO_RSA ${lib_path}src/exemple/crypto_RSA.cpp)
endif()

set(COUNT_ALL_FILE ${lib_path}src/exemple/count_all_file.cpp)

if (OPENCV_FOUND)
    set(IMAGE_DIFF ${lib_path}src/exemple/image_diff.cpp)
    set(LIST_ALL_FILE_OPENCV ${lib_path}src/exemple/list_all_file_opencv.cpp)
endif()

#   ====================================================
#   ====================    LIBS    ====================
#   ====================================================

#set(MAIN_LLIBS Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Quick Qt${QT_VERSION_MAJOR}::Multimedia sfml-graphics sfml-window sfml-audio sfml-network sfml-system ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${OpenCV_LIBS})
#set(MAIN_LLIBS sfml-graphics sfml-window sfml-audio sfml-network sfml-system ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${OpenCV_LIBS})

set(MAIN_LLIBS "")

if(Boost_FOUND)
    set(MAIN_LLIBS ${MAIN_LLIBS} ${Boost_LIBRARIES})
endif()

if(SFML_FOUND)
    set(MAIN_LLIBS ${MAIN_LLIBS} sfml-graphics sfml-window sfml-audio sfml-network sfml-system)
endif()

if (OPENCV_FOUND)
    set(MAIN_LLIBS ${MAIN_LLIBS} ${OpenCV_LIBS})
endif()

if (OPENGL_FOUND)
    set(MAIN_LLIBS ${MAIN_LLIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${CUDA_LIBRARY})
endif()

if (QT_FOUND)
    set(MAIN_LLIBS ${MAIN_LLIBS} Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Quick Qt${QT_VERSION_MAJOR}::Multimedia)
endif()

#   ==================================================
#   ===============    EXEC FILES    =================
#   ==================================================

add_executable(bench_isprime ${BENCH_ISPRIME})
target_link_libraries(bench_isprime PRIVATE thread Threads::Threads vector math time)
add_dependencies(bench_isprime vector math time thread)

add_executable(bench_vector_copy ${BENCH_VECTOR_COPY})
target_link_libraries(bench_vector_copy PRIVATE thread Threads::Threads vector math time)
add_dependencies(bench_vector_copy thread vector math time)

add_executable(bench_vector_sort ${BENCH_VECTOR_SORT})
target_link_libraries(bench_vector_sort PRIVATE thread Threads::Threads vector math time)
add_dependencies(bench_vector_sort thread vector math time)

add_executable(bench_digits_count ${BENCH_DIGITS_COUNT})
target_link_libraries(bench_digits_count PRIVATE thread Threads::Threads vector math time)
add_dependencies(bench_digits_count thread vector math time)

add_executable(bench_find_max ${BENCH_FIND_MAX})
target_link_libraries(bench_find_max PRIVATE thread Threads::Threads vector math time)
add_dependencies(bench_find_max thread vector math time)

add_executable(bench_pc ${BENCH_PC})
target_link_libraries(bench_pc PRIVATE time vector)
add_dependencies(bench_pc time vector)

add_executable(bench_crc32 ${BENCH_CRC32})
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_crc32 PRIVATE thread Threads::Threads vector my_crypto math time OpenMP::OpenMP_CXX)
else()
    target_link_libraries(bench_crc32 PRIVATE thread Threads::Threads vector my_crypto math time)
endif()
add_dependencies(bench_crc32 thread Threads::Threads vector my_crypto math time)


if(Boost_FOUND)
    add_executable(gta_sa_cheat ${GTA_SA_CHEAT})
    target_link_libraries(gta_sa_cheat PRIVATE string_lib time OpenMP::OpenMP_CXX)
    add_dependencies(gta_sa_cheat string_lib time OpenMP::OpenMP_CXX)

    add_executable(list_all_file ${LIST_ALL_FILE})
    target_link_libraries(list_all_file PRIVATE ${Boost_LIBRARIES} filesystem)
endif()

if(OpenCL_FOUND)
    add_executable(opencl_test ${OPENCL_TEST})
    target_link_libraries(opencl_test PRIVATE ${OpenCL_LIBRARY})

    add_executable(opencl_test2 ${OPENCL_TEST2})
    target_link_libraries(opencl_test2 PRIVATE ${OpenCL_LIBRARY})

    add_executable(opencl_findstringinv ${OPENCL_FINDSTRINGINV})
    target_link_libraries(opencl_findstringinv PRIVATE ${OpenCL_LIBRARY} string_lib)

    add_executable(opencl_gta_sa_gpu ${OPENCL_GTA_SA_GPU})
    target_link_libraries(opencl_gta_sa_gpu PRIVATE ${OpenCL_LIBRARY} string_lib)

    if (OPENSSL_FOUND)
        add_executable(opencl_crc32 ${OPENCL_CRC32})
        target_link_libraries(opencl_crc32 PRIVATE ${OpenCL_LIBRARY} my_crypto time)
    endif()
endif()

add_executable(parse_csv ${PARSE_CSV})
target_link_libraries(parse_csv PRIVATE string_lib)
add_dependencies(parse_csv string_lib)

if (OPENSSL_FOUND)
    add_executable(list_all_file_hash ${LIST_ALL_FILE_HASH})
    target_link_libraries(list_all_file_hash PRIVATE ${Boost_LIBRARIES} ${OPENSSL_LIBRARIES} my_crypto OpenMP::OpenMP_CXX time)
    add_dependencies(list_all_file_hash my_crypto OpenMP::OpenMP_CXX time)

    add_executable(crypto_AES ${CRYPTO_AES})
    target_link_libraries(crypto_AES PRIVATE ${Boost_LIBRARIES} ${OPENSSL_LIBRARIES} filesystem my_crypto thread time)
    add_dependencies(crypto_AES filesystem thread time)

    add_executable(crypto_RSA ${CRYPTO_RSA})
    target_link_libraries(crypto_RSA PRIVATE ${Boost_LIBRARIES} ${OPENSSL_LIBRARIES} filesystem my_crypto thread time)
    add_dependencies(crypto_RSA my_crypto thread time filesystem)
endif()

if (OPENCV_FOUND)
    add_executable(list_all_file_opencv ${LIST_ALL_FILE_OPENCV})
    target_link_libraries(list_all_file_opencv PRIVATE filesystem ${OpenCV_LIBS})
    add_dependencies(list_all_file_opencv filesystem ${OpenCV_LIBS})

    add_executable(image_diff ${IMAGE_DIFF})
    target_link_libraries(image_diff PRIVATE filesystem ${OpenCV_LIBS} utils_opencv)
endif()

#   ================================================
#   ================    TESTING     ================
#   ================================================
option(ENABLE_TESTS "Enable tests" ON)

if(Boost_FOUND AND ENABLE_TESTS)
    enable_testing()
    #Test getSchwarzschild
    add_executable(test_getSchwarzschild ${lib_path}src/test/math/test_getSchwarzschild.cpp)
    target_link_libraries(test_getSchwarzschild PRIVATE ${Boost_LIBRARIES})
    add_test(NAME test_getSchwarzschild COMMAND ${CMAKE_COMMAND} --build . --target test_getSchwarzschild)

    #Test getGravitationalAttraction
    add_executable(test_getGravitationalAttraction ${lib_path}src/test/math/test_getGravitationalAttraction.cpp)
    target_link_libraries(test_getGravitationalAttraction PRIVATE ${Boost_LIBRARIES})
    add_test(NAME test_getGravitationalAttraction COMMAND test_getGravitationalAttraction)

    #Test sphere
    add_executable(test_sphere ${lib_path}src/test/math/test_sphere.cpp)
    target_link_libraries(test_sphere PRIVATE ${Boost_LIBRARIES} quadmath)
    add_test(NAME test_sphere COMMAND test_sphere)

    #Test cylinder
    add_executable(test_cylinder ${lib_path}src/test/math/test_cylinder.cpp)
    target_link_libraries(test_cylinder PRIVATE ${Boost_LIBRARIES})
    add_test(NAME test_cylinder COMMAND test_cylinder)

    #Test vector sort
    add_executable(test_vector_sort ${lib_path}src/test/vector/test_vector_sort.cpp)
    target_link_libraries(test_vector_sort PRIVATE ${Boost_LIBRARIES} vector)
    add_test(NAME test_vector_sort COMMAND test_vector_sort)

    #Test vector copy
    add_executable(test_vector_copy ${lib_path}src/test/vector/test_vector_copy.cpp)
    target_link_libraries(test_vector_copy PRIVATE ${Boost_LIBRARIES} vector)
    add_test(NAME test_vector_copy COMMAND test_vector_copy)

    #Test vector copy
    add_executable(test_vector_max_simd ${lib_path}src/test/vector/test_vector_max_simd.cpp)
    target_link_libraries(test_vector_max_simd PRIVATE ${Boost_LIBRARIES} vector)
    add_test(NAME test_vector_max_simd COMMAND test_vector_max_simd)

    add_executable(test_string_seq ${lib_path}src/test/string/test_string_seq.cpp)
    target_link_libraries(test_string_seq PRIVATE ${Boost_LIBRARIES} string_lib)
    add_test(NAME test_string_seq COMMAND test_string_seq)
endif()

if (OPENSSL_FOUND AND Boost_FOUND AND ENABLE_TESTS)
    #Test crypto
    add_executable(test_crypto ${lib_path}src/test/crypto/test_crypto.cpp)
    target_link_libraries(test_crypto PRIVATE ${Boost_LIBRARIES} ${OPENSSL_LIBRARIES} my_crypto)
    add_test(NAME test_crypto COMMAND test_crypto)

    #Test crypto AES
    add_executable(test_crypto_aes ${lib_path}src/test/crypto/test_crypto_AES.cpp)
    target_link_libraries(test_crypto_aes PRIVATE ${Boost_LIBRARIES} ${OPENSSL_LIBRARIES} my_crypto)
    add_test(NAME test_crypto_aes COMMAND test_crypto_aes)

    add_executable(test_crypto_CRC ${lib_path}src/test/crypto/test_crypto_CRC.cpp)
    target_link_libraries(test_crypto_CRC PRIVATE ${Boost_LIBRARIES} my_crypto)
    add_test(NAME test_crypto_CRC COMMAND test_crypto_CRC)
endif()

#Test C string
if(Boost_FOUND AND ENABLE_TESTS)
    
    set(TEST_REV_STR test_rev_str)
    add_executable(${TEST_REV_STR} ${lib_path}src/test/C/string_c/test_string.cpp)
    target_link_libraries(${TEST_REV_STR} PRIVATE ${Boost_LIBRARIES} string_c)
    add_test(NAME ${TEST_REV_STR} COMMAND ${TEST_REV_STR})

    set(TEST_IS_NUM test_my_isnum)
    add_executable(${TEST_IS_NUM} ${lib_path}src/test/C/string_is/test_my_isnum.cpp)
    target_link_libraries(${TEST_IS_NUM} PRIVATE ${Boost_LIBRARIES} string_is_c)
    add_test(NAME ${TEST_IS_NUM} COMMAND ${TEST_IS_NUM})

    set(TEST_STR_IS_NUM test_my_str_isnum)
    add_executable(${TEST_STR_IS_NUM} ${lib_path}src/test/C/string_is/test_my_str_isnum.cpp)
    target_link_libraries(${TEST_STR_IS_NUM} PRIVATE ${Boost_LIBRARIES} string_is_c)
    add_test(NAME ${TEST_STR_IS_NUM} COMMAND ${TEST_STR_IS_NUM})


    #Test power
    set(TEST_POWER test_power)
    add_executable(${TEST_POWER} ${lib_path}src/test/math/test_power.cpp)
    target_link_libraries(${TEST_POWER} PRIVATE ${Boost_LIBRARIES})
    add_test(NAME ${TEST_POWER} COMMAND ${TEST_POWER})
endif()
#add_test(NAME ${TEST_POWER} COMMAND ${CMAKE_COMMAND} --build . --target ${TEST_POWER})

#add_custom_command(
#     TARGET test_power
#     COMMENT "Run tests"
#     POST_BUILD 
#     WORKING_DIRECTORY .
#     COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failures --extra-verbose
#)

#   ================================================
#   ===============    COPY FILES    ===============
#   ================================================

#file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/screenshot)

# For OpenCL_test.cpp
configure_file(${lib_path}lib/opencl/kernels/vector_add_kernel.cl ${CMAKE_BINARY_DIR}/kernels/vector_add_kernel.cl COPYONLY)
# For OpenCL_test2.cpp
configure_file(${lib_path}lib/opencl/kernels/simple_add.cl ${CMAKE_BINARY_DIR}/kernels/simple_add.cl COPYONLY)
# For OpenCL_crc32.cpp
configure_file(${lib_path}lib/opencl/kernels/crc_kernel.cl ${CMAKE_BINARY_DIR}/kernels/crc_kernel.cl COPYONLY)
# For OpenCL_findStringInv.cpp
configure_file(${lib_path}lib/opencl/kernels/findStringInv.cl ${CMAKE_BINARY_DIR}/kernels/findStringInv.cl COPYONLY)
# For GTA SA Alternative cheats
configure_file(${lib_path}lib/opencl/kernels/gta_sa.cl ${CMAKE_BINARY_DIR}/kernels/gta_sa.cl COPYONLY)

#add_custom_command(
#    TARGET game
#    COMMENT "CP Texture"
#    POST_BUILD 
#    WORKING_DIRECTORY .
#    COMMAND cp -R -u -p "../texture/PIPOYA FREE RPG Character Sprites 32x32" "texture/PIPOYA FREE RPG Character Sprites 32x32"
#)

#   =================================================
#   ===============    UNSET VALUE    ===============
#   =================================================


#unset(CMAKE_BUILD_TYPE CACHE)

#   ================================================
#   =================    CPack    ==================
#   ================================================

include(InstallRequiredSystemLibraries)

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "My funky project")
set(CPACK_PACKAGE_VENDOR "Me, myself, and I")
set(CPACK_PACKAGE_NAME "BenLib")
set(CPACK_PACKAGE_CONTACT "bensuperpc@gmail.com")
#set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt")
#set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/Copyright.txt")
#set(CPACK_PACKAGE_VERSION_MAJOR "1")
#set(CPACK_PACKAGE_VERSION_MINOR "3")
#set(CPACK_PACKAGE_VERSION_PATCH "2")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DIRECTORY "Package")

set(CPACK_PACKAGE_INSTALL_DIRECTORY "CMake ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")
if(WIN32 AND NOT UNIX)
  set(CPACK_PACKAGE_ICON "${CMake_SOURCE_DIR}/Utilities/Release\\\\InstallIcon.bmp")
  set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\MyExecutable.exe")
  set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} My Famous Project")
  set(CPACK_NSIS_HELP_LINK "http:\\\\\\\\www.bensuperpc.org")
  set(CPACK_NSIS_URL_INFO_ABOUT "http:\\\\\\\\www.bensuperpc.com")
  set(CPACK_NSIS_CONTACT "${CPACK_PACKAGE_CONTACT}")
  set(CPACK_NSIS_MODIFY_PATH ON)
else()
  set(CPACK_STRIP_FILES "bin/MyExecutable")
  set(CPACK_SOURCE_STRIP_FILES "")
endif()

set(CPACK_PACKAGE_RELEASE "${CMAKE_BUILD_TYPE}")
set(CPACK_PACKAGE_EXECUTABLES "MyExecutable" "My Executable")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")

# ====== RPM ======
set(CPACK_RPM_PACKAGE_GROUP "vesoft")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_PACKAGE_RELOCATABLE ON)
set(CPACK_RPM_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}")
set(CPACK_RPM_PACKAGE_RELEASE ON)
set(CPACK_RPM_PACKAGE_DESCRIPTION "${CPACK_PACKAGE_DESCRIPTION_SUMMARY}")
set(CPACK_RPM_PACKAGE_AUTOREQ ON)

set(CPACK_RPM_PACKAGE_REQUIRES "clang >= 10.0.0, cmake >= 3.0")
set(CPACK_RPM_PACKAGE_SUGGESTS "")
set(CPACK_RPM_PACKAGE_CONFLICTS "libxml2")

#set(CPACK_RPM_PACKAGE_SUGGESTS "clang >= 10.0.0, cmake >= 3.0")
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/home")
list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/home/vesoft")
#set(CPACK_RPM_COMPRESSION_TYPE "xz")
set(CPACK_RPM_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.rpm")

# ====== DEB ======
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Bensuperpc")
set(CPACK_DEBIAN_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")

set(CPACK_DEBIAN_PACKAGE_DEPENDS "clang (>= 10.0.0), libc6 (>= 3.0)")
set(CPACK_DEBIAN_PACKAGE_BREAKS "")
set(CPACK_DEBIAN_PACKAGE_CONFLICTS "")
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "")

set(CPACK_DEBIAN_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.deb")
set(CPACK_DEB_COMPONENT_INSTALL OFF)
set(CPACK_DEBIAN_COMPRESSION_TYPE "lzma")
set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION FALSE)


# ====== LIBS ======
#install(TARGETS my_crypto
#  ARCHIVE
#  DESTINATION lib
#  COMPONENT libraries)
#install(TARGETS my_cuda
#  ARCHIVE
#  DESTINATION lib
#  COMPONENT libraries)
install(TARGETS math
  ARCHIVE
  DESTINATION lib
  COMPONENT libraries)
install(TARGETS string_lib
  RUNTIME
  DESTINATION lib
  COMPONENT libraries)
install(TARGETS convert_utils
  RUNTIME
  DESTINATION lib
  COMPONENT libraries)
install(TARGETS texture
  RUNTIME
  DESTINATION lib
  COMPONENT libraries)
install(TARGETS utils_opencv
  RUNTIME
  DESTINATION lib
  COMPONENT libraries)
install(TARGETS utils_opengl
  RUNTIME
  DESTINATION lib
  COMPONENT libraries)
install(TARGETS time
  RUNTIME
  DESTINATION lib
  COMPONENT libraries)
install(TARGETS cpu
  RUNTIME
  DESTINATION lib
  COMPONENT libraries)
install(TARGETS vector
  RUNTIME
  DESTINATION lib
  COMPONENT libraries)
install(TARGETS thread
  RUNTIME
  DESTINATION lib
  COMPONENT libraries)
install(TARGETS filesystem
  RUNTIME
  DESTINATION lib
  COMPONENT libraries)
install(TARGETS std
  RUNTIME
  DESTINATION lib
  COMPONENT libraries)
# ====== BIN ======
install(TARGETS list_all_file_hash
  RUNTIME
  DESTINATION bin
  COMPONENT applications)
# ====== TESTS ======
install(TARGETS test_power
  RUNTIME
  DESTINATION tests
  COMPONENT tests)
#install(FILES mylib.h
#  DESTINATION include
#  COMPONENT headers)

set(CPACK_COMPONENTS_ALL applications libraries headers tests)
set(CPACK_GENERATOR "TZ;STGZ;TGZ;RPM;ZIP;7Z;DEB")
# Section CPack
include (CPack)
