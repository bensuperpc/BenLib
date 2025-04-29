cmake_minimum_required(VERSION 3.14)

project(benlibExamples CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)

if(PROJECT_IS_TOP_LEVEL)
  find_package(benlib REQUIRED)
endif()

add_subdirectory(../external/raylib external/raylib)
add_subdirectory(../external/raylib-cpp external/raylib-cpp)

add_custom_target(run-examples)

function(add_example NAME)
  add_executable("${NAME}" "${NAME}.cpp")
  target_link_libraries("${NAME}" PRIVATE benlib::benlib)
  target_link_libraries("${NAME}" PRIVATE raylib raylib_cpp)
  target_compile_features("${NAME}" PRIVATE cxx_std_23)
  add_custom_target("run_${NAME}" COMMAND "${NAME}" VERBATIM)
  add_dependencies("run_${NAME}" "${NAME}")
  add_dependencies(run-examples "run_${NAME}")
endfunction()

add_example(basic_anti_cheat)
add_example(pattern_repository)
add_example(bench_example)
add_example(request_exemple)
add_example(raylib_test)

add_folders(Example)
