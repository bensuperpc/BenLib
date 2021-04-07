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
#  Source:                                                   #
#  OS: ALL                                                   #
#  CPU: ALL                                                  #
#                                                            #
##############################################################



# https://bugs.archlinux.org/task/62258
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

#export PATH=/opt/cuda/bin:$PATH
#export LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

#set(CUTLASS_NATIVE_CUDA_INIT ON)
#CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES
#CMAKE_CUDA_COMPILER
#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) 

# Can fix some issues
if(NOT CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda;/usr;/usr/local/cuda")
endif()

find_package(CUDA 10.0 QUIET)

if (CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS} ${BLAS_INCLUDE_DIRS} ${CUDA_CUBLAS_DIRS})
    link_directories(${CUDA_LIBRARIES} ${BLAS_LIBRARIES})
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    #set(CMAKE_CUDA_FLAGS	"${CMAKE_CUDA_FLAGS}	-Xcompiler=-Wall")
    #set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")
#–default-stream per-thread
    message(STATUS "CUDA: FOUND on ${PROJECT_NAME}")
    message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
else()
    message(STATUS "CUDA: NOT FOUND on ${PROJECT_NAME}")
    find_package(CUDA 10.0)
endif()



#set_target_properties(BENLIB PROPERTIES CUDA_SEPARABLE_COMPILATION ON)



function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
    get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
    if(NOT "${old_flags}" STREQUAL "")
        string(REPLACE ";" "," CUDA_flags "${old_flags}")
        set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
            "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
            )
    endif()
endfunction()