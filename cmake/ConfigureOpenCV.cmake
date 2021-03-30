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
#  Created: 29, March, 2021                                  #
#  Modified: 29, March, 2021                                 #
#  file: ConfigureBoost.cmake                                #
#  CMake                                                     #
#  Source:  https://mirkokiefer.com/cmake-by-example-f95eb47d45b1                                                   #
#  OS: ALL                                                   #
#  CPU: ALL                                                  #
#                                                            #
##############################################################


if (WIN64)
    set(OpenCV_DIR "C:/Users/Benoit/Downloads/opencv-4.4.0/build")
elseif(ANDROID)
    set(OpenCV_DIR "~/android/OpenCV-android-sdk/sdk/native/jni" )
endif()

find_package(OpenCV REQUIRED QUIET)

if (OPENCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    #add_compile_definitions(OPENCV_VERSION="${OpenCV_VERSION}")
    message(STATUS "OPENCV: FOUND")
else()
    message(STATUS "OPENCV: NOT FOUND")
    #ExternalProject_Add(opencv_lib
    #    GIT_REPOSITORY https://github.com/opencv/opencv
    #    GIT_TAG 4.4.0
    #    SOURCE_DIR opencv
    #    BINARY_DIR opencv-build
    #    CMAKE_ARGS -D CMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    #    CMAKE_ARGS -D BUILD_EXAMPLES:BOOL=OFF -D OPENCV_ENABLE_NONFREE:BOOL=ON -D BUILD_TESTS:BOOL=OFF
    #    CMAKE_ARGS -D INSTALL_C_EXAMPLES:BOOL=OFF -D INSTALL_PYTHON_EXAMPLES=OFF
        #CMAKE_ARGS -D OPENCV_EXTRA_MODULES_PATH=/build/opencv_contrib-4.1.1/modules
        #CMAKE_ARGS -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        #-D WITH_CUDA=ON
    #)
    #set( OPENCV_ROOT_DIR ${EXTERNAL_INSTALL_LOCATION})
    #set( OPENCV_DIR ${EXTERNAL_INSTALL_LOCATION})
    #set(CPACK_INSTALL_CMAKE_PROJECTS "${CPACK_INSTALL_CMAKE_PROJECTS};${Dep_DIR};Dep;ALL;/")

    #ExternalProject_Get_Property(opencv_lib install_dir)
endif() 
