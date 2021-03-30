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

set_target_properties(${PROJECT_NAME} PROPERTIES
    PUBLIC_HEADER "${HEADERS}"
)

install(TARGETS ${PROJECT_NAME} 
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # Nedd more 1 target
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Nedd more 1 target
#    PRIVATE_HEADER DESTINATION source
    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}"
)
