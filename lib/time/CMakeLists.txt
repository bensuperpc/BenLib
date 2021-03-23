cmake_minimum_required(VERSION 3.11)

project(utils)

set(SRCS
	date/date.cpp
	chrono/chrono.cpp
	)
	
set(HEADERS
	date/date.hpp
	chrono/chrono.hpp
	)
	

#add_library(time_static STATIC ${SRCS} ${HEADERS})
add_library(time SHARED ${SRCS} ${HEADERS})
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set_target_properties(time PROPERTIES UNITY_BUILD ON)
	#target_precompile_headers(time PUBLIC ${HEADERS})
    message(STATUS "CMAKE_UNITY_BUILD : 1 on ${PROJECT_NAME}")
else()
    message(STATUS "CMAKE_UNITY_BUILD : 0 on ${PROJECT_NAME}")
endif()