cmake_minimum_required(VERSION 3.11)

project(utils CXX C)

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
