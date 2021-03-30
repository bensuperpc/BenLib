cmake_minimum_required(VERSION 3.10)

project(hello C)

set(SRCS
	hello.c
	)
	
set(HEADERS
	hello.h
	)

#add_library(hello_static STATIC ${SRCS} ${HEADERS})
add_library(hello SHARED ${SRCS} ${HEADERS})

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
