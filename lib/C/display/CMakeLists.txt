cmake_minimum_required(VERSION 3.10)

project(string_c C)

set(SRCS
	my_putstr.c
	my_putchar.c
	)
	
set(HEADERS
	my_putstr.h
	my_putchar.h
	)

#add_library(display_c_static STATIC ${SRCS} ${HEADERS})
add_library(display_c SHARED ${SRCS} ${HEADERS})

#install (TARGETS hello DESTINATION bin)
#install (FILES hello.h DESTINATION include)
