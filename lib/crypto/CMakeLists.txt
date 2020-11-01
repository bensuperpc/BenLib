cmake_minimum_required(VERSION 3.11)

project(my_crypto C CXX)

set(SRCS
	)
	
set(HEADERS
	)

#add_library(math_static STATIC ${SRCS} ${HEADERS})
add_library(my_crypto SHARED ${SRCS} ${HEADERS})

#install (TARGETS math DESTINATION bin)
#install (FILES math.h DESTINATION include)
