cmake_minimum_required(VERSION 3.11)

project(math VERSION 0.0.5 LANGUAGES CXX C)


set(SRCS
	getGravitationalAttraction_imp.hpp
	getSchwarzschild_imp.hpp
	sphere_imp.hpp
	cylinder_imp.hpp
	power_imp.hpp
	power.c
	pair.cpp
	sphere_imp.hpp
	square_root.cpp
	prime.cpp
	float.cpp
	random.cpp
	)
	
set(HEADERS
	getGravitationalAttraction.hpp
	getSchwarzschild.hpp
	sphere.hpp
	cylinder.hpp
	constant.hpp
	power.hpp
	power.h
	pair.hpp
	square_root.hpp
	prime.hpp
	prime_imp.hpp
	count_digits_imp.hpp
	count_digits.hpp
	float.hpp
	random.hpp
	)

#add_library(math_static STATIC ${SRCS} ${HEADERS})
add_library(math SHARED ${SRCS} ${HEADERS})
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set_target_properties(math PROPERTIES UNITY_BUILD ON)
	#target_precompile_headers(math PUBLIC ${HEADERS})
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
