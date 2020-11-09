cmake_minimum_required(VERSION 3.10)

project(string_c C)

set(SRCS
	my_revstr.c
	my_strcpy.c
	my_strdup.c
	my_strlen.c
	my_strllen.c
	)
	
set(HEADERS
	my_revstr.h
    my_strcpy.h
	my_strdup.h
	my_strlen.h
	my_strllen.h
	)

#add_library(string_c_static STATIC ${SRCS} ${HEADERS})
add_library(string_c SHARED ${SRCS} ${HEADERS})

#install (TARGETS hello DESTINATION bin)
#install (FILES hello.h DESTINATION include)
