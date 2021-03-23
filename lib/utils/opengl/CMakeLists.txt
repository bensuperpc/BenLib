cmake_minimum_required(VERSION 3.11)

project(utils LANGUAGES CXX C)

set(SRCS
	screen_save.cpp
	screen_save.c
	)
	
set(HEADERS
	screen_save.hpp
	screen_save.h
	)

#add_library(utilsutils_opengl_static STATIC ${SRCS} ${HEADERS})
add_library(utils_opengl SHARED ${SRCS} ${HEADERS})
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set_target_properties(utils_opengl PROPERTIES UNITY_BUILD ON)
	#target_precompile_headers(utils_opengl PUBLIC ${HEADERS})
    message(STATUS "CMAKE_UNITY_BUILD : 1 on ${PROJECT_NAME}")
else()
    message(STATUS "CMAKE_UNITY_BUILD : 0 on ${PROJECT_NAME}")
endif()