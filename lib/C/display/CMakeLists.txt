cmake_minimum_required(VERSION 3.10)

project(display_c C)

set(SRCS
	my_putstr.c
	my_putchar.c
	)
	
set(HEADERS
	my_putstr.h
	my_putchar.h
	)

#add_library(display_c_static STATIC ${SRCS} ${HEADERS})
add_library(display_c ${SRCS} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES
    PUBLIC_HEADER "${HEADERS}"
)
#SHARED
install(TARGETS ${PROJECT_NAME} 
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # Nedd more 1 target
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Nedd more 1 target
#    PRIVATE_HEADER DESTINATION source
    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}"
)
