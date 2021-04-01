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
#  Created: 01, April, 2021                                  #
#  Modified: 01, April, 2021                                 #
#  file: CMakeLists.txt                                      #
#  CMake                                                     #
#  Source:  
#  OS: ALL                                                   #
#  CPU: ALL                                                  #
#                                                            #
##############################################################

cmake_minimum_required(VERSION 3.12)
project(my_arduino LANGUAGES CXX C)


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

set(SRCS
    led/blink_number.cpp
    reboot/reboot_software.cpp
    usefull/arduino_type.cpp
    usefull/free_ram.cpp
    usefull/voltage.cpp
    arduino_serial.cpp
	)
	
set(HEADERS
    led/blink_number.hpp
    reboot/reboot_software.hpp
    usefull/arduino_type.hpp
    usefull/free_ram.hpp
    usefull/voltage.hpp
    arduino_compatibility.hpp
    arduino_serial.hpp
    )


#set_target_properties(my_cuda PROPERTIES PUBLIC_HEADER "${EDV_INCLUDE_FILES}")

include_directories("${PROJECT_SOURCE_DIR}")

add_library(my_arduino SHARED ${SRCS} ${HEADERS})
set_target_properties(my_arduino PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
set_target_properties(${PROJECT_NAME} PROPERTIES
    PUBLIC_HEADER "${HEADERS}"
)

# https://github.com/awakecoding/cmake/blob/master/Tests/Complex/Executable/CMakeLists.txt
set_source_files_properties(tests/basic.ino PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY 1)
#set_target_properties(tests/basic.ino PROPERTIES LINKER_LANGUAGE CXX)

add_executable(test_arduino tests/basic.cpp tests/basic.ino)
#target_link_libraries(test1 PRIVATE )
#set_property(TARGET vec1d PROPERTY CUDA_ARCHITECTURES 52 60 61 62 70 72 75)
set_target_properties(test_arduino PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
target_compile_definitions(test_arduino PRIVATE FILE_ARG=tests/basic.ino)



install(TARGETS ${PROJECT_NAME} 
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # Nedd more 1 target
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Nedd more 1 target
#    PRIVATE_HEADER DESTINATION source
    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}"
)
