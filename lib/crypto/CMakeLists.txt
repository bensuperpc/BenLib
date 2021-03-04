cmake_minimum_required(VERSION 3.11)

project(my_crypto C CXX)

find_package(OpenSSL QUIET)

if (OPENSSL_FOUND)
    include_directories( ${OPENSSL_INCLUDE_DIR})
    link_directories(${OPENSSL_LIBRARIES})
    message(STATUS "OPENSSL: FOUND on my_crypto")
else()
    message(STATUS "OPENSSL: NOT FOUND on my_crypto")
endif()


set(SRCS
    crypto.cpp
    crypto_AES.cpp
    crypto_RSA.cpp
    crypto_CRC32.cpp
	)
	
set(HEADERS
    crypto.hpp
    crypto_AES.hpp
    crypto_RSA.hpp
    crypto_CRC32.hpp
    )

#add_library(my_crypto_static STATIC ${SRCS} ${HEADERS})
add_library(my_crypto SHARED ${SRCS} ${HEADERS})
target_link_libraries(my_crypto PRIVATE ssl crypto)