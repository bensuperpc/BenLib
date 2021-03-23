cmake_minimum_required(VERSION 3.11)

project(cpu VERSION 0.0.2 LANGUAGES CXX C)

set(SRCS
	cpu.cpp
	)
	
set(HEADERS
	cpu.hpp
	)

#add_library(math_static STATIC ${SRCS} ${HEADERS})
add_library(cpu SHARED ${SRCS} ${HEADERS})
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set_target_properties(cpu PROPERTIES UNITY_BUILD ON)
	target_precompile_headers(cpu PUBLIC ${HEADERS})
    message(STATUS "CMAKE_UNITY_BUILD : 1 on ${PROJECT_NAME}")
else()
    message(STATUS "CMAKE_UNITY_BUILD : 0 on ${PROJECT_NAME}")
endif()
#install (TARGETS math DESTINATION bin)
#install (FILES math.h DESTINATION include)
