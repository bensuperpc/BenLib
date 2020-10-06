cmake_minimum_required(VERSION 3.10)

project(string_c C)

set(SRCS
	revstr.c
	)
	
set(HEADERS
	revstr.h
	)

#add_library(string_c_static STATIC ${SRCS} ${HEADERS})
add_library(string_c SHARED ${SRCS} ${HEADERS})

#install (TARGETS hello DESTINATION bin)
#install (FILES hello.h DESTINATION include)
