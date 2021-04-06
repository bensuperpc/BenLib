#
# CMakeLists.txt - make build
#
# Created by Bensuperpc at 18, July of 2020
# Modified by Bensuperpc at 30, March of 2021
#
# Released into the Public domain with MIT licence
# https://opensource.org/licenses/MIT
# MIT
#
# Written with Kate 20.04.3
# Script compatibility : Linux (Manjaro and ArchLinux)
#
# ==============================================================================

cmake_minimum_required (VERSION 3.10)


if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
# Top-level project
else()
# Sub-level project
endif()

project(python_binding LANGUAGES CXX C)


if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # Top-level project
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} -g3 -Og -ggdb3")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -march=native -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} -g3 -Og -ggdb3")
else()
# Sub-level project
endif()


#Find boost lib
find_package(Boost COMPONENTS python REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

#Find python
#set(PYTHON_LIBRARY "\\usr\\lib\\python2.7")
#set(PYTHON_INCLUDE_DIR "\\usr\\include\\python2.7")
find_package(PythonLibs 3.8 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIR})

# Include header directory
include_directories("header")

set(SRCS
    src/function_binding.c
	)
	
set(HEADERS
    header/function_binding.h
    )


#Build libs
add_library(class_binding SHARED src/class_binding.cpp)
set_target_properties(class_binding PROPERTIES PREFIX "") #Remove lib prefix
target_link_libraries(class_binding ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

add_library(function_binding SHARED src/function_binding.cpp)
set_target_properties(function_binding PROPERTIES PREFIX "") #Remove lib prefix
target_link_libraries(function_binding ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

add_library(function_binding_c SHARED ${SRCS} ${HEADERS})
set_target_properties(function_binding_c PROPERTIES PREFIX "") #Remove lib prefix
target_link_libraries(function_binding_c ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

configure_file(python/class_binding.py ${CMAKE_BINARY_DIR}/bin/class_binding.py COPYONLY)
configure_file(python/function_binding.py ${CMAKE_BINARY_DIR}/bin/function_binding.py COPYONLY)
configure_file(python/function_binding_c.py ${CMAKE_BINARY_DIR}/bin/function_binding_c.py COPYONLY)

set_target_properties(function_binding_c PROPERTIES
    PUBLIC_HEADER "${HEADERS}"
)

install(TARGETS function_binding
#    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # Nedd more 1 target
#    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Nedd more 1 target
#    PRIVATE_HEADER DESTINATION source
#    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}"
)

install(TARGETS function_binding_c
#    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # Nedd more 1 target
#    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Nedd more 1 target
#    PRIVATE_HEADER DESTINATION source
    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}"
)

install(TARGETS class_binding
#    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # Nedd more 1 target
#    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Nedd more 1 target
#    PRIVATE_HEADER DESTINATION source
#    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}"
)

# Copy script
install(FILES
    python/class_binding.py
    python/function_binding.py
    python/function_binding_c.py
    DESTINATION bin
)