cmake_minimum_required(VERSION 3.10)

project(libhello C)

set(SRCS
	hello.c
	)
	
set(HEADERS
	hello.h
	)

#add_library(hello_static STATIC ${SRCS} ${HEADERS})
add_library(hello SHARED ${SRCS} ${HEADERS})

#install (TARGETS hello DESTINATION bin)
#install (FILES hello.h DESTINATION include)
