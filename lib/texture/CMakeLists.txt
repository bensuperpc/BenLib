cmake_minimum_required(VERSION 3.11)

project(texture CXX C)

set(SRCS
	sfml_screenshot.cpp
	load_titlemap.cpp
	load_texture.cpp
	load_texturemap.cpp
	load_title.cpp
	texture.cpp
	)
	
set(HEADERS
	sfml_screenshot.hpp
	load_titlemap.hpp
	load_texture.hpp
	load_texturemap.hpp
	load_title.hpp
	texture.hpp
	)
	

#add_library(texture_static STATIC ${SRCS} ${HEADERS})
add_library(texture ${SRCS} ${HEADERS})
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set_target_properties(texture PROPERTIES UNITY_BUILD ON)
	#target_precompile_headers(texture PUBLIC ${HEADERS})
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