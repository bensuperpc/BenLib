cmake_minimum_required(VERSION 3.11)

project(utils CXX C)

set(SRCS
	opencv_utils.cpp
	)
	
set(HEADERS
	opencv_utils.hpp
	)

#add_library(utils_opencv_static STATIC ${SRCS} ${HEADERS})
add_library(utils_opencv SHARED ${SRCS} ${HEADERS})
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set_target_properties(utils_opencv PROPERTIES UNITY_BUILD ON)
	target_precompile_headers(utils_opencv PUBLIC ${HEADERS})
    message(STATUS "CMAKE_UNITY_BUILD : 1 on ${PROJECT_NAME}")
else()
    message(STATUS "CMAKE_UNITY_BUILD : 0 on ${PROJECT_NAME}")
endif()