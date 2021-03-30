cmake_minimum_required(VERSION 3.10)

project(pointer_c C)

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
