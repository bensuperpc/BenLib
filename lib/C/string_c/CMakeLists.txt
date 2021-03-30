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
