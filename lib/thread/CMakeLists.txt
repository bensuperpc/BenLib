cmake_minimum_required(VERSION 3.11)

project(thread)

set(SRCS
	Pool.hpp
	printer.hpp
	ThreadPool.h
	)
	
set(HEADERS
	Pool.cpp
	printer.cpp
	)
	

#add_library(thread STATIC ${SRCS} ${HEADERS})
add_library(thread SHARED ${SRCS} ${HEADERS})
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set_target_properties(thread PROPERTIES UNITY_BUILD ON)
	#target_precompile_headers(thread PUBLIC ${HEADERS})
    message(STATUS "CMAKE_UNITY_BUILD : 1 on ${PROJECT_NAME}")
else()
    message(STATUS "CMAKE_UNITY_BUILD : 0 on ${PROJECT_NAME}")
endif()