cmake_minimum_required(VERSION 3.11)

project(filesystem VERSION 0.0.2 LANGUAGES CXX C)

if (WIN32)
    set(Boost_USE_STATIC_LIBS TRUE)
    set(Boost_USE_MULTITHREADED TRUE)
    set(Boost_USE_STATIC_RUNTIME FALSE)
else ()
    set(Boost_USE_STATIC_LIBS FALSE)
    set(Boost_USE_MULTITHREADED TRUE)
    set(Boost_USE_STATIC_RUNTIME FALSE)
endif (WIN32)

find_package(Boost 1.65.0 COMPONENTS thread filesystem system unit_test_framework QUIET)

if(Boost_FOUND)
    message(STATUS "Boost FOUND on ${PROJECT_NAME}")
    include_directories( ${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARIES})
else()
    message(STATUS "Boost: NOT FOUND on ${PROJECT_NAME}")
endif()

set(SRCS
	filesystem.cpp
	)
	
set(HEADERS
	filesystem.hpp
	)
	

#add_library(filesystem_static STATIC ${SRCS} ${HEADERS})
add_library(filesystem SHARED ${SRCS} ${HEADERS})
target_link_libraries(filesystem ${Boost_LIBRARIES})
