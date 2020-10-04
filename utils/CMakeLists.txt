cmake_minimum_required(VERSION 3.11)

project(utils CXX C)

set(SRCS
	#sfml/screen_save.cpp
	opengl/screen_save.cpp
	opengl/screen_save.c
	opencv/opencv_utils.cpp
	)
	
set(HEADERS
	#sfml/screen_save.hpp
	opengl/screen_save.hpp
	opengl/screen_save.h
	opencv/opencv_utils.hpp
	)

#add_library(utils_static STATIC ${SRCS} ${HEADERS})
add_library(utils SHARED ${SRCS} ${HEADERS})
