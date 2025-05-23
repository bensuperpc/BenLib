cmake_minimum_required(VERSION 3.14)

project(benlibTests LANGUAGES C CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)

# ---- Dependencies ----

if(PROJECT_IS_TOP_LEVEL)
  find_package(benlib REQUIRED)
  enable_testing()
endif()


function(test_bench_generator TEST_BENCH_NAME IS_TEST ADD_TO_TEST)

  set(SOURCES
    )

  set(HEADERS
    )

  if (IS_TEST)
    add_executable("${TEST_BENCH_NAME}" "source/test/${TEST_BENCH_NAME}.cpp" ${SOURCES} ${HEADERS})
  else()
    add_executable("${TEST_BENCH_NAME}" "source/benchmark/${TEST_BENCH_NAME}.cpp" ${SOURCES} ${HEADERS})
  endif()


  if (IS_TEST)
    target_link_libraries("${TEST_BENCH_NAME}" PRIVATE gtest)
  else()
    target_link_libraries("${TEST_BENCH_NAME}" PRIVATE benchmark::benchmark)
  endif()

  target_link_libraries("${TEST_BENCH_NAME}" PRIVATE benlib::benlib)
  #target_link_libraries("${TEST_BENCH_NAME}" PRIVATE raylib)
  #target_link_libraries("${TEST_BENCH_NAME}" PRIVATE FastNoise2)
  #target_link_libraries("${TEST_BENCH_NAME}" PRIVATE spdlog::spdlog nlohmann_json::nlohmann_json)

  if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
    target_link_libraries("${TEST_BENCH_NAME}" PRIVATE OpenMP::OpenMP_CXX)
  endif()

  set_target_properties("${TEST_BENCH_NAME}"
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

  #if(NOT CMAKE_BUILD_TYPE MATCHES Debug AND NOT CMAKE_BUILD_TYPE MATCHES Coverage)
  #  add_test(NAME "${TEST_BENCH_NAME}" COMMAND $<TARGET_FILE:${TEST_BENCH_NAME}>)
  #elseif()
  #  message(STATUS "Disable ${BENCH_NAME}, Performance benchmark test only run on Release/RelWithDebInfo/MinSizeRel")
  #endif()

  if (ADD_TO_TEST)
      add_test(NAME "${TEST_BENCH_NAME}" COMMAND $<TARGET_FILE:${TEST_BENCH_NAME}>)
  endif()
  target_compile_features("${TEST_BENCH_NAME}" PRIVATE cxx_std_23)
endfunction()

# ---- Tests ----

if(NOT WIN32)
    #include(../cmake/lib/gtest.cmake)
    #include(../cmake/lib/benchmark.cmake)
    #include(../cmake/lib/openmp.cmake)
    #include(../cmake/lib/raygui.cmake)
    add_subdirectory(../external/googletest external/googletest)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_TESTING OFF)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_FORCE_WERROR OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_CXX_LINKER_FLAGS "" CACHE STRING "" FORCE)
    set(BENCHMARK_CXX_LIBRARIES "" CACHE STRING "" FORCE)
    set(BENCHMARK_CXX_FLAGS "" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_COVERAGE "" CACHE STRING "" FORCE)
    set(CMAKE_REQUIRED_FLAGS "" CACHE STRING "" FORCE)
    add_subdirectory(../external/benchmark external/benchmark)

    #include(../cmake/lib/raylib.cmake)
    #include(../cmake/lib/fast_noise2.cmake)
    #include(../cmake/lib/spdlog.cmake)
    #include(../cmake/lib/json.cmake)
    include(../cmake/utile/sccache.cmake)

    include_directories(source)

    #test_bench_generator(benlib_bench false false)


    test_bench_generator(getSchwarzschild_test true true)
    test_bench_generator(getGravitationalAttraction_test true true)
    test_bench_generator(cylinder_test true true)
    test_bench_generator(power_test true true)
    test_bench_generator(sphere_test true true)
    test_bench_generator(angle_test true true)
    test_bench_generator(float_test true true)
    test_bench_generator(random_test true true)
    test_bench_generator(keyword_test true true)
    test_bench_generator(gettersetter_test true true)
    test_bench_generator(scheduler_test true true)
endif()

# ---- End-of-file commands ----

add_folders(Test)
