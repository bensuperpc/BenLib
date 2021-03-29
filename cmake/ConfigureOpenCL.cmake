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