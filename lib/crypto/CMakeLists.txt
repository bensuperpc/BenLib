cmake_minimum_required(VERSION 3.11)

project(my_crypto VERSION 0.0.2 LANGUAGES CXX C)

find_package(OpenSSL QUIET)

if (OPENSSL_FOUND)
    include_directories( ${OPENSSL_INCLUDE_DIR})
    link_directories(${OPENSSL_LIBRARIES})
    message(STATUS "OPENSSL: FOUND on ${PROJECT_NAME}")
else()
    message(STATUS "OPENSSL: NOT FOUND on ${PROJECT_NAME}")
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
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set_target_properties(my_crypto PROPERTIES UNITY_BUILD ON)
	#target_precompile_headers(my_crypto PUBLIC ${HEADERS})
    message(STATUS "CMAKE_UNITY_BUILD : 1 on ${PROJECT_NAME}")
else()
    message(STATUS "CMAKE_UNITY_BUILD : 0 on ${PROJECT_NAME}")
endif()

install(TARGETS my_crypto
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)