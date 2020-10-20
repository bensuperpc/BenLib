cmake_minimum_required(VERSION 3.11)

project(texture)

set(SRCS
	filesystem.cpp
	)
	
set(HEADERS
	filesystem.hpp
	)
	

#add_library(filesystem_static STATIC ${SRCS} ${HEADERS})
add_library(filesystem SHARED ${SRCS} ${HEADERS})
