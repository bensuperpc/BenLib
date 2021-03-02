cmake_minimum_required(VERSION 3.11)

project(string_lib)

set(SRCS
	parsing.cpp
	string_lib.cpp
	)
	
set(HEADERS
	parsing.hpp
	string_lib.hpp
	)
	

#add_library(filesystem_static STATIC ${SRCS} ${HEADERS})
add_library(string_lib SHARED ${SRCS} ${HEADERS})
