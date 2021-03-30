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
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set_target_properties(filesystem PROPERTIES UNITY_BUILD ON)
	target_precompile_headers(filesystem PUBLIC ${HEADERS})
    message(STATUS "CMAKE_UNITY_BUILD : 1 on ${PROJECT_NAME}")
else()
    message(STATUS "CMAKE_UNITY_BUILD : 0 on ${PROJECT_NAME}")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    PUBLIC_HEADER "${HEADERS}"
)

install(TARGETS ${PROJECT_NAME} 
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # Nedd more 1 target
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Nedd more 1 target
#    PRIVATE_HEADER DESTINATION source
    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}"
)