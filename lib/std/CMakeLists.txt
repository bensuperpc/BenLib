cmake_minimum_required(VERSION 3.11)

project(std)

set(SRCS
std.cpp
	)
	
set(HEADERS
	std.hpp
	)
	

#add_library(filesystem_static STATIC ${SRCS} ${HEADERS})
add_library(std SHARED ${SRCS} ${HEADERS})
