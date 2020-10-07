cmake_minimum_required(VERSION 3.11)

project(utils CXX C)

set(SRCS
	opencv_utils.cpp
	)
	
set(HEADERS
	opencv_utils.hpp
	)

#add_library(utils_opencv_static STATIC ${SRCS} ${HEADERS})
add_library(utils_opencv SHARED ${SRCS} ${HEADERS})
