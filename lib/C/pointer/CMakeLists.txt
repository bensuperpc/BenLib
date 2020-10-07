cmake_minimum_required(VERSION 3.10)

project(string_c C)

set(SRCS
	my_strcpy_ptr.c
	my_inv_two_str_pt.c
	my_revstr_pt.c
	)
	
set(HEADERS
	my_revstr_pt.h
	my_strcpy_ptr.h
	my_inv_two_str_pt.h
	)

#add_library(pointer_c_static STATIC ${SRCS} ${HEADERS})
add_library(pointer_c SHARED ${SRCS} ${HEADERS})

#install (TARGETS hello DESTINATION bin)
#install (FILES hello.h DESTINATION include)
