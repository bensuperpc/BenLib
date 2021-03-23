cmake_minimum_required(VERSION 3.11)

project(vector C CXX)

set(SRCS
	vector.cpp
	vector_avx.cpp
	)
	
set(HEADERS
	vector.hpp
	vector_imp.hpp
	vector_avx.hpp
	)

#add_library(math_static STATIC ${SRCS} ${HEADERS})
add_library(vector SHARED ${SRCS} ${HEADERS})
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set_target_properties(vector PROPERTIES UNITY_BUILD ON)
	target_precompile_headers(vector PUBLIC ${HEADERS})
    message(STATUS "CMAKE_UNITY_BUILD : 1 on ${PROJECT_NAME}")
else()
    message(STATUS "CMAKE_UNITY_BUILD : 0 on ${PROJECT_NAME}")
endif()

#install (TARGETS math DESTINATION bin)
#install (FILES math.h DESTINATION include)
