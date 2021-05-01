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

#add_library(string_is_c_static STATIC ${SRCS} ${HEADERS}) SHARED
add_library(string_is_c ${SRCS} ${HEADERS})

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
