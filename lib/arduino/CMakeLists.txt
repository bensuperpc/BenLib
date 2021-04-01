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

 #set_source_files_properties(foo.bar PROPERTIES LANGUAGE CXX)
 #add_executable(foo foo.bar)
 #set_target_properties(foo PROPERTIES LINKER_LANGUAGE CXX)

#set_target_properties(my_cuda PROPERTIES PUBLIC_HEADER "${EDV_INCLUDE_FILES}")

include_directories("${PROJECT_SOURCE_DIR}")

add_library(my_arduino SHARED ${SRCS} ${HEADERS})
set_target_properties(my_arduino PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

#set(EDV_PUBLIC_INCLUDE_DIRECTORIES      kernel/)
#set(EDV_PRIVATE_INCLUDE_DIRECTORIES     kernel/)
#target_include_directories(${PROJECT_NAME} PUBLIC ${EDV_PUBLIC_INCLUDE_DIRECTORIES})
#target_include_directories(${PROJECT_NAME} PRIVATE ${EDV_PRIVATE_INCLUDE_DIRECTORIES})

#file(GLOB_RECURSE EDV_INCLUDE_FILES "kernel/*.hpp*")
#file(GLOB_RECURSE EDV_SOURCE_FILES "kernel/*.cpp*")

set_target_properties(${PROJECT_NAME} PROPERTIES
    PUBLIC_HEADER "${HEADERS}"
)

install(TARGETS ${PROJECT_NAME} 
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # Nedd more 1 target
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Nedd more 1 target
#    PRIVATE_HEADER DESTINATION source
    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}"
)
