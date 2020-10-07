cmake_minimum_required(VERSION 3.11)

project(thread)

set(SRCS
	Pool.hpp
	printer.hpp
	ThreadPool.h
	)
	
set(HEADERS
	Pool.cpp
	printer.cpp
	)
	

#add_library(thread STATIC ${SRCS} ${HEADERS})
add_library(thread SHARED ${SRCS} ${HEADERS})
