cmake_minimum_required(VERSION 3.10)

project(string_c C)

set(SRCS
	my_compute_square_root.c
	my_sort_int_array.c
	my_compute_power_rec.c
	my_count_spe_char.c
	my_swap.c
	my_is_prime.c
	)
	
set(HEADERS
	my_compute_square_root.h
	my_sort_int_array.h
	my_compute_power_rec.h
	my_count_spe_char.h
	my_swap.h
	my_is_prime.h
	)

#add_library(integer_c_static STATIC ${SRCS} ${HEADERS})
add_library(integer_c SHARED ${SRCS} ${HEADERS})

#install (TARGETS hello DESTINATION bin)
#install (FILES hello.h DESTINATION include)
