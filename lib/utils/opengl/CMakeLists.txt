cmake_minimum_required(VERSION 3.11)

project(utils_opengl LANGUAGES CXX C)

set(SRCS
	screen_save.cpp
	screen_save.c
	)
	
set(HEADERS
	screen_save.hpp
	screen_save.h
	)

#add_library(utilsutils_opengl_static STATIC ${SRCS} ${HEADERS})
add_library(utils_opengl ${SRCS} ${HEADERS})
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set_target_properties(utils_opengl PROPERTIES UNITY_BUILD ON)
	#target_precompile_headers(utils_opengl PUBLIC ${HEADERS})
    message(STATUS "CMAKE_UNITY_BUILD : 1 on ${PROJECT_NAME}")
else()
    message(STATUS "CMAKE_UNITY_BUILD : 0 on ${PROJECT_NAME}")
endif()

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