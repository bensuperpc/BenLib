#
# CMakeLists.txt - make build
#
# Created by jwdinius
# Modified by Bensuperpc at 18, July of 2020
#
# Released into the Public domain with MIT licence
# https://opensource.org/licenses/MIT
# MIT
#
# Written with Kate 20.04.3
# Script compatibility : Linux (Manjaro and ArchLinux)
#
# ==============================================================================

cmake_minimum_required (VERSION 3.10)
project(python_binding LANGUAGES CXX C)

#Find boost lib
find_package(Boost REQUIRED COMPONENTS python)
include_directories(${Boost_INCLUDE_DIR})

#Find python
#set(PYTHON_LIBRARY "\\usr\\lib\\python2.7")
#set(PYTHON_INCLUDE_DIR "\\usr\\include\\python2.7")
find_package(PythonLibs 3.8 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIR})

#Build libs
add_library(class_binding SHARED src/class_binding.cpp)
#set_target_properties(class_binding PROPERTIES PREFIX "") #Remove lib prefix
target_link_libraries(class_binding ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

add_library(function_binding SHARED src/function_binding.cpp)
#set_target_properties(function_binding PROPERTIES PREFIX "") #Remove lib prefix
target_link_libraries(function_binding ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
#add_library(function_binding_c SHARED src/function_binding.c)
#set_target_properties(function_binding_c PROPERTIES PREFIX "") #Remove lib prefix
#target_link_libraries(function_binding_c ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

#Install
#install(TARGETS pyHello DESTINATION /usr/lib/python3.8/site-packages)

#set_target_properties(${PROJECT_NAME} PROPERTIES
#    PUBLIC_HEADER "${HEADERS}"
#)

install(TARGETS class_binding
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # Nedd more 1 target
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Nedd more 1 target
#    PRIVATE_HEADER DESTINATION source
    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}"
)

install(TARGETS function_binding
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # Nedd more 1 target
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Nedd more 1 target
#    PRIVATE_HEADER DESTINATION source
    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}"
)
