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