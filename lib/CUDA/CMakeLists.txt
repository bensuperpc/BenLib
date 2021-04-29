##############################################################
#   ____                                                     #
#  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___    #
#  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __|   #
#  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__    #
#  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___|   #
#                             |_|             |_|            #
##############################################################
#                                                            #
#  BenLib, 2021                                              #
#  Created: 16, March, 2021                                  #
#  Modified: 17, March, 2021                                 #
#  file: CMakeLists.txt                                      #
#  CMake                                                     #
#  Source:  https://developer.nvidia.com/blog/building-cuda-applications-cmake/                                                   #
#           https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
#           https://github.com/robertmaynard/code-samples/blob/master/posts/cmake/CMakeLists.txt
#           https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/
#           https://stackoverflow.com/questions/33442355/cmake-file-for-c-cuda-project
#           https://stackoverflow.com/questions/58861467/how-to-compile-c-as-cuda-using-cmake
#           https://gist.github.com/ashwin/6547060
#           https://stackoverflow.com/questions/61056517/build-cuda-kernel-with-meson
#           https://mesonbuild.com/Tutorial.html
#           https://cmake.org/cmake/help/git-stage/prop_tgt/CUDA_ARCHITECTURES.html
#           https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
#           https://github.com/onqtam/awesome-cmake
#           https://hothardware.com/news/cuda-on-intel-gpus-zluda
#           https://stackoverflow.com/questions/18273760/cmake-how-to-add-cuda-to-existing-project
#           https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9444-build-systems-exploring-modern-cmake-cuda-v2.pdf
#           https://gitlab.kitware.com/cmake/cmake/-/issues/16586
#           https://releases.llvm.org/6.0.0/docs/CompileCudaWithLLVM.html
#           https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
#           https://stackoverflow.com/questions/53256405/howto-pass-flag-to-nvcc-compiler-in-cmake
#           https://github.com/Kitware/CMake/blob/master/Tests/Cuda/Complex/CMakeLists.txt
#  OS: ALL                                                   #
#  CPU: ALL                                                  #
#                                                            #
##############################################################

cmake_minimum_required(VERSION 3.12)
#project(my_cuda LANGUAGES CUDA CXX C)
project(my_cuda LANGUAGES CUDA CXX C)


#include(CheckLanguage)
#check_language(CUDA)
#if(CMAKE_CUDA_COMPILER)
#	enable_language(CUDA)
#    message(STATUS "CMAKE_CUDA_COMPILER: True")
#endif()


if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # Top-level project
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} -g3 -Og -ggdb3")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -march=native -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} -g3 -Og -ggdb3")
else()
    # Sub-level project
endif()

#   ================    CUDA    ================
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

# Can fix some issues
#if(NOT CUDA_TOOLKIT_ROOT_DIR)
#    set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda;/usr;/usr/local/cuda")
#endif()
if(NOT CMAKE_TOOLCHAIN_FILE)
    find_package(CUDA 10.0 QUIET)

    if (CUDA_FOUND)
        include_directories(${CUDA_INCLUDE_DIRS} ${BLAS_INCLUDE_DIRS} ${CUDA_CUBLAS_DIRS})
        link_directories(${CUDA_LIBRARIES} ${BLAS_LIBRARIES})
        if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
            message(STATUS "CMAKE_CUDA_STANDARD: C++17")
        set(CMAKE_CUDA_STANDARD 17)
        else()
            message(STATUS "CMAKE_CUDA_STANDARD: C++14")
            set(CMAKE_CUDA_STANDARD 14)
        endif()
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        #set(CMAKE_CUDA_FLAGS	"${CMAKE_CUDA_FLAGS}	-Xcompiler=-Wall")
        #set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")
    #–default-stream per-thread
        message(STATUS "CUDA: FOUND on ${PROJECT_NAME}")
        message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "CUDA: ${CUDA_INCLUDE_DIRS} at ${CUDA_CUDART_LIBRARY}")
    else()
        message(STATUS "CUDA: NOT FOUND on ${PROJECT_NAME}")
    endif()
else()
    include_directories("${CUDA_INCLUDE_DIRS}")
    message("set(CUDA_TOOLKIT_ROOT_DIR \"${CUDA_TOOLKIT_ROOT_DIR}\")")
    message("set(CUDA_INCLUDE_DIRS \"${CUDA_INCLUDE_DIRS}\")")
    message("set(CUDA_TOOLKIT_INCLUDE \"${CUDA_TOOLKIT_INCLUDE}\")")
    message("set(CUDA_CUDART_LIBRARY \"${CUDA_CUDART_LIBRARY}\")")
    message("set(CUDA_cublas_LIBRARY \"${CUDA_cublas_LIBRARY}\")")
    message("set(CUDA_LIBRARIES \"${CUDA_LIBRARIES}\")")
    message("set(CUDA_cudart_static_LIBRARY \"${CUDA_cudart_static_LIBRARY}\")")
    message("set(CUDA_cudadevrt_LIBRARY \"${CUDA_cudadevrt_LIBRARY}\")")
    message("set(CUDA_NVCC_EXECUTABLE \"${CUDA_NVCC_EXECUTABLE}\")")
    message("set(CUDA_SDK_ROOT_DIR \"${CUDA_SDK_ROOT_DIR}\")")
    message("set(CUDA_SDK_SEARCH_PATH \"${CUDA_SDK_SEARCH_PATH}\")")
    message("set(CUDA_cufft_LIBRARY \"${CUDA_cufft_LIBRARY}\")")
    message("set(CUDA_nppc_LIBRARY \"${CUDA_nppc_LIBRARY}\")")
    message("set(CUDA_nppial_LIBRARY \"${CUDA_nppial_LIBRARY}\")")
    message("set(CUDA_nppicc_LIBRARY \"${CUDA_nppicc_LIBRARY}\")")
    message("set(CUDA_nppicom_LIBRARY \"${CUDA_nppicom_LIBRARY}\")")
    message("set(CUDA_nppidei_LIBRARY \"${CUDA_nppidei_LIBRARY}\")")
    message("set(CUDA_nppif_LIBRARY \"${CUDA_nppif_LIBRARY}\")")
    message("set(CUDA_nppig_LIBRARY \"${CUDA_nppig_LIBRARY}\")")
    message("set(CUDA_nppim_LIBRARY \"${CUDA_nppim_LIBRARY}\")")
    message("set(CUDA_nppist_LIBRARY \"${CUDA_nppist_LIBRARY}\")")
    message("set(CUDA_nppisu_LIBRARY \"${CUDA_nppisu_LIBRARY}\")")
    message("set(CUDA_nppitc_LIBRARY \"${CUDA_nppitc_LIBRARY}\")")
    message("set(CUDA_npps_LIBRARY \"${CUDA_npps_LIBRARY}\")")
    message("set(CMAKE_C_COMPILER \"${CMAKE_C_COMPILER}\")")
    message("set(CMAKE_CXX_COMPILER \"${CMAKE_CXX_COMPILER}\")")
    message("set(CMAKE_CUDA_COMPILER \"${CMAKE_CUDA_COMPILER}\")")
endif()

if(CUDA_NVCC_EXECUTABLE AND NOT CUDA_VERSION)
  # Compute the version.
  execute_process (COMMAND ${CUDA_NVCC_EXECUTABLE} "--version" OUTPUT_VARIABLE NVCC_OUT)
  string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR ${NVCC_OUT})
  string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR ${NVCC_OUT})
  set(CUDA_VERSION "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" CACHE STRING "Version of CUDA as computed from nvcc.")
  message("CUDA version: ${CUDA_VERSION}")
else()
  # Need to set these based off of the cached value
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR "${CUDA_VERSION}")
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR "${CUDA_VERSION}")
endif()

#For clang CUDA
# --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}
# -lcudart_static or -lcudart
# -x cuda
# --cuda-gpu-arch=sm_30
# Optional
# CUDACXX=clang++
# 

#   ================    OpenMP    ================
find_package(OpenMP QUIET)

if (OPENMP_FOUND)
    message(STATUS "OPENMP: FOUND on ${PROJECT_NAME}")
    #set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    #set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    #set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(STATUS "OPENMP: NOT FOUND on ${PROJECT_NAME}")
endif()


set(SRCS
    kernel/kernel.cu
    kernel/matrix.cu
    kernel/crc.cu
    kernel/cuda_utils.cpp
	)
	
set(HEADERS
    kernel/kernel.cuh
    kernel/kernel.h
    kernel/kernel.hpp
    kernel/matrix.cuh
    kernel/matrix.h
    kernel/matrix.hpp
    kernel/matrix.tpp
    kernel/matrix_ops.hpp
    kernel/matrix_ops.tpp
    kernel/crc.cuh
    kernel/crc.h
    kernel/crc.hpp
    kernel/cuda_utils.hpp
    kernel/cuda_utils.h
    )

#cuda_add_library(my_cuda STATIC ${SRCS} ${HEADERS} OPTIONS -arch sm_75)
add_library(my_cuda SHARED ${SRCS} ${HEADERS})
#cuda_add_executable

#set_target_properties(my_cuda PROPERTIES PUBLIC_HEADER "${EDV_INCLUDE_FILES}")

include_directories("kernel/")

#if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
#    target_compile_features(my_cuda PUBLIC cxx_std_17)
#else()
#    target_compile_features(my_cuda PUBLIC cxx_std_14)
#endif()

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ")
    
if(CUDA_VERSION VERSION_GREATER_EQUAL "11.2")
    target_compile_options(my_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --threads 0 -Wall -Wextra -pipe #--std=c++14
        -Xptxas -O2 --default-stream per-thread
        -Xcompiler=-Wall -Xcompiler=-fopenmp -Xcompiler=-Wextra
        -Xcompiler -flto
        #--resource-usage
        # Maxwell
        -arch=sm_52
        -gencode=arch=compute_52,code=sm_52
        -gencode arch=compute_53,code=sm_53
        # Pascale
        -gencode=arch=compute_60,code=sm_60
        -gencode=arch=compute_61,code=sm_61
        -gencode=arch=compute_62,code=sm_62
        # Turing
        -gencode=arch=compute_70,code=sm_70
        -gencode=arch=compute_72,code=sm_72
        -gencode=arch=compute_75,code=compute_75
        # Ampere 80 and 86
        >)
    #target_compile_options(my_cuda PRIVATE "-fno-lto") # Disable due issues
endif()

if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
    target_compile_options(my_cuda PRIVATE "-fno-lto")
endif()

set_target_properties(my_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(my_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(my_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# If support unity build, improve speed on headers
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
	#target_precompile_headers(convert_utils PUBLIC ${HEADERS})
endif()
set_property(TARGET my_cuda PROPERTY CUDA_ARCHITECTURES 52 60 61 62 70 72 75)
set_target_properties(my_cuda PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
#target_link_libraries(my_cuda PRIVATE ${CUDA_LIBRARIES})

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET my_cuda 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

if(UNIX)
  # Help the shared cuda runtime find libcudart as it is not located
  # in a default system searched location
  set_property(TARGET my_cuda PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

add_executable(vec1d vec1d.c)
target_link_libraries(vec1d PRIVATE ${CUDA_LIBRARIES} my_cuda m)
set_property(TARGET vec1d PROPERTY CUDA_ARCHITECTURES 52 60 61 62 70 72 75)
set_target_properties(vec1d PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

add_executable(vec2d_mult vec2d_mult.c)
if (OPENMP_FOUND)
    target_link_libraries(vec2d_mult PRIVATE ${CUDA_LIBRARIES} OpenMP::OpenMP_C my_cuda m)
else()
    target_link_libraries(vec2d_mult PRIVATE ${CUDA_LIBRARIES} my_cuda m)
endif()
set_property(TARGET vec2d_mult PROPERTY CUDA_ARCHITECTURES 52 60 61 62 70 72 75)
set_target_properties(vec2d_mult PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

add_executable(vec2d_add vec2d_add.c)
if (OPENMP_FOUND)
    target_link_libraries(vec2d_add PRIVATE ${CUDA_LIBRARIES} OpenMP::OpenMP_C my_cuda m)
else()
    target_link_libraries(vec2d_add PRIVATE ${CUDA_LIBRARIES} my_cuda)
endif()
set_property(TARGET vec2d_add PROPERTY CUDA_ARCHITECTURES 52 60 61 62 70 72 75)
set_target_properties(vec2d_add PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

add_executable(vec2d_mult2 vec2d_mult.cpp)
if (OPENMP_FOUND)
    target_link_libraries(vec2d_mult2 PRIVATE ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX my_cuda m)
else()
    target_link_libraries(vec2d_mult2 PRIVATE ${CUDA_LIBRARIES} my_cuda)
endif()
set_property(TARGET vec2d_mult2 PROPERTY CUDA_ARCHITECTURES 52 60 61 62 70 72 75)
set_target_properties(vec2d_mult2 PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

add_executable(vec2d_mult3 vec2d_mult3.cpp)
if (OPENMP_FOUND)
    target_link_libraries(vec2d_mult3 PRIVATE ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX my_cuda m)
else()
    target_link_libraries(vec2d_mult3 PRIVATE ${CUDA_LIBRARIES} my_cuda)
endif()
set_property(TARGET vec2d_mult3 PROPERTY CUDA_ARCHITECTURES 52 60 61 62 70 72 75)
set_target_properties(vec2d_mult3 PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

add_executable(vec3d_mult vec3d_mult.cpp)
if (OPENMP_FOUND)
    target_link_libraries(vec3d_mult PRIVATE ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX my_cuda m)
else()
    target_link_libraries(vec3d_mult PRIVATE ${CUDA_LIBRARIES} my_cuda)
endif()
set_property(TARGET vec3d_mult PROPERTY CUDA_ARCHITECTURES 52 60 61 62 70 72 75)
set_target_properties(vec3d_mult PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

add_executable(vec3d_mult2 vec3d_mult2.cpp)
if (OPENMP_FOUND)
    target_link_libraries(vec3d_mult2 PRIVATE ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX my_cuda m)
else()
    target_link_libraries(vec3d_mult2 PRIVATE ${CUDA_LIBRARIES} my_cuda m)
endif()
set_property(TARGET vec3d_mult2 PROPERTY CUDA_ARCHITECTURES 52 60 61 62 70 72 75)
set_target_properties(vec3d_mult2 PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)


set(EDV_PUBLIC_INCLUDE_DIRECTORIES      kernel/)
set(EDV_PRIVATE_INCLUDE_DIRECTORIES     kernel/)
target_include_directories(${PROJECT_NAME} PUBLIC ${EDV_PUBLIC_INCLUDE_DIRECTORIES})
target_include_directories(${PROJECT_NAME} PRIVATE ${EDV_PRIVATE_INCLUDE_DIRECTORIES})

#file(GLOB_RECURSE EDV_INCLUDE_FILES "kernel/*.hpp*")
#file(GLOB_RECURSE EDV_SOURCE_FILES "kernel/*.cpp*")

set_target_properties(${PROJECT_NAME} PROPERTIES
    PUBLIC_HEADER "${HEADERS}"
)

install(TARGETS ${PROJECT_NAME} vec1d vec2d_add vec2d_mult vec2d_mult2 vec2d_mult3 vec3d_mult vec3d_mult2
#   INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}"
#    CONFIGURATIONS Release
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#    PRIVATE_HEADER DESTINATION source
    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}"
)

#install(TARGETS vec1d
#  RUNTIME
#  DESTINATION bin
#  COMPONENT applications)


#${CMAKE_INSTALL_INCLUDEDIR}
#install(TARGETS my_cuda
#        CONFIGURATIONS Debug
#        RUNTIME DESTINATION Debug/bin)
#install(TARGETS my_cuda
#        CONFIGURATIONS Release
#        RUNTIME DESTINATION Release/bin)

#install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/kernel/matrix.tpp"
#  DESTINATION include
#  )
#
#install(
#    DIRECTORY ${CMAKE_SOURCE_DIR}/include/
#    DESTINATION include
#    FILES_MATCHING PATTERN "*.h*")

