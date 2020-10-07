cmake_minimum_required(VERSION 3.11)

project(utils)

set(SRCS
	date/date.cpp
	chrono/chrono.cpp
	)
	
set(HEADERS
	date/date.hpp
	chrono/chrono.hpp
	)
	

#add_library(time_static STATIC ${SRCS} ${HEADERS})
add_library(time SHARED ${SRCS} ${HEADERS})
