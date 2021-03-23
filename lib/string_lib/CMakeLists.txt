cmake_minimum_required(VERSION 3.11)

project(string_lib)

set(SRCS
	parsing.cpp
	string_lib.cpp
	)
	
set(HEADERS
	parsing.hpp
	string_lib.hpp
	string_lib_impl.hpp
	)
	

#add_library(filesystem_static STATIC ${SRCS} ${HEADERS})
add_library(string_lib SHARED ${SRCS} ${HEADERS})
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set_target_properties(string_lib PROPERTIES UNITY_BUILD ON)
	target_precompile_headers(string_lib PUBLIC ${HEADERS})
    message(STATUS "CMAKE_UNITY_BUILD : 1 on ${PROJECT_NAME}")
else()
    message(STATUS "CMAKE_UNITY_BUILD : 0 on ${PROJECT_NAME}")
endif()