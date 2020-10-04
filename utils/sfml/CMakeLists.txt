cmake_minimum_required(VERSION 3.11)

project(utils CXX C)

set(SRCS
	screen_save.cpp
	)
	
set(HEADERS
	screen_save.hpp
	)

#add_library(utils_sfml_static STATIC ${SRCS} ${HEADERS})
add_library(utils_sfml SHARED ${SRCS} ${HEADERS})
