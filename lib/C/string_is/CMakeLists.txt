cmake_minimum_required(VERSION 3.10)

project(string_is_c C)

set(SRCS
	my_isnum.c
	my_str_isnum.c
	my_str_isalpha.c
	my_str_islower.c
	my_str_isupper.c
	my_str_isprintable.c
	)
	
set(HEADERS
	my_isnum.h
	my_str_isnum.h
	my_str_isalpha.h
	my_str_islower.h
	my_str_isupper.h
	my_str_isprintable.h
	)

#add_library(string_is_c_static STATIC ${SRCS} ${HEADERS})
add_library(string_is_c SHARED ${SRCS} ${HEADERS})

#install (TARGETS hello DESTINATION bin)
#install (FILES hello.h DESTINATION include)
