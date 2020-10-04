cmake_minimum_required(VERSION 3.11)

project(cpu C CXX)

set(SRCS
	cpu.cpp
	)
	
set(HEADERS
	cpu.hpp
	)

#add_library(math_static STATIC ${SRCS} ${HEADERS})
add_library(cpu SHARED ${SRCS} ${HEADERS})

#install (TARGETS math DESTINATION bin)
#install (FILES math.h DESTINATION include)
