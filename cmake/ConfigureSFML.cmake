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

if(NOT DEFINED SFML_ROOT AND WIN32)
    set(SFML_ROOT "C:\\Project Files (x86)\\SFML")
endif()
#set(SFML_USE_STATIC_STD_LIBS 0)
#set(SFML_STATIC_LIBRARIES 0)
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