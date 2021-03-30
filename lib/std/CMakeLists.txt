cmake_minimum_required (VERSION 3.1 FATAL_ERROR)
project(std VERSION 0.0.5 DESCRIPTION "Add make_unique_ptr on C++11" LANGUAGES CXX)

project(std)

set(SRCS
std.cpp
	)
	
set(HEADERS
	std.hpp
	)
	

#add_library(filesystem_static STATIC ${SRCS} ${HEADERS})
add_library(std SHARED ${SRCS} ${HEADERS})
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set_target_properties(std PROPERTIES UNITY_BUILD ON)
	#target_precompile_headers(std PUBLIC ${HEADERS})
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