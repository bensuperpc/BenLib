cmake_minimum_required(VERSION 3.11)

project(my_crypto C CXX)

find_package(OpenSSL QUIET)

if (OPENSSL_FOUND)
    include_directories( ${OPENSSL_INCLUDE_DIR})
    link_directories(${OPENSSL_LIBRARIES})
    message(STATUS "OPENSSL: FOUND")
else()
    message(STATUS "OPENSSL: NOT FOUND")
endif()


set(SRCS
    crypto.cpp
	)
	
set(HEADERS
    crypto.hpp
    )

#add_library(math_static STATIC ${SRCS} ${HEADERS})
add_library(my_crypto SHARED ${SRCS} ${HEADERS})
target_link_libraries(my_crypto PRIVATE ssl crypto)