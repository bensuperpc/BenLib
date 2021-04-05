##############################################################
#   ____                                                     #
#  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___    #
#  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __|   #
#  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__    #
#  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___|   #
#                             |_|             |_|            #
##############################################################
#                                                            #
#  BenLib, 2020                                              #
#  Created: 05, February, 2020                               #
#  Modified: 29, March, 2021                                 #
#  file: CMakeLists.txt                                      #
#  CMake                                                     #
#  Source:  https://medium.com/@mikkelwilson/opencv-with-cuda-acceleration-test-e9f7155e1c84                                                   #
#           https://onqtam.com/programming/2019-12-20-pch-unity-cmake-3-16/
#           https://stackoverflow.com/questions/61548932/force-cmake-to-rebuild-precompiled-headers
#           https://www.virag.si/2015/07/use-ccache-with-cmake-for-faster-compilation/
#           https://cmake.org/cmake/help/latest/module/CPack.html#variable:CPACK_BINARY_%3CGENNAME%3E
#           https://cmake.org/cmake/help/v3.0/module/CPack.html
#           https://gitlab.kitware.com/cmake/community/-/wikis/doc/cpack/PackageGenerators
#           https://cmake.org/cmake/help/v3.0/module/CPackRPM.html
#           https://blog.usejournal.com/creating-debian-packages-cmake-e519a0186e87
#           https://stackoverflow.com/questions/49644273/cmake-change-stack-size
#           https://stackoverflow.com/questions/46713143/cmake-install-library-and-development-files-as-separate-components-using-public
#           https://github.com/Pseudomanifold/cmake-cpp
#           https://cmake.org/cmake/help/v3.1/prop_tgt/C_EXTENSIONS.html#prop_tgt:C_EXTENSIONS
#           https://stackoverflow.com/a/60203765/10152334
#           https://github.com/StableCoder/cmake-scripts#sanitizer-builds-sanitizerscmake
#           http://www.stablecoder.ca/2018/01/15/code-coverage.html
#           https://down.firefliestudio.com/Linux/leveldb/cmake-3.19.0-rc3/Tests/FindDoxygen/StampFile/CMakeLists.txt
#           https://discourse.cmake.org/t/doxygen-add-docs-install-output-directory/2904
#           https://antonypeacock.blog/2018/11/03/adding-doxygen-support-to-a-cmake-project/
#           https://github.com/nCine/nCine/blob/master/cmake/ncine_installation.cmake
#           
#  OS: ALL                                                   #
#  CPU: ALL                                                  #
#                                                            #
##############################################################

cmake_minimum_required (VERSION 3.12 FATAL_ERROR)
project(BENLIB VERSION 0.0.5 DESCRIPTION "Bensuperpc's libs" LANGUAGES CXX C HOMEPAGE_URL "https://github.com/Bensuperpc/BenLib")

#Add CUDA if it found
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
    message(STATUS "CMAKE_CUDA_COMPILER: True")
endif()

#set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#   ==================================================
#   ================    CMAKE CONF    ================
#   ==================================================

#   ================    CMAKE MODULE    ================
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake)

#"Whether to use ccache to speed up compiling"
option(ENABLE_CCACHE ON "If you want enable CCACHE, default: ON")
# Include
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Increase verbose CMake
set(CMAKE_VERBOSE_MAKEFILE ON)
# Have cmake create a compile database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#Disable precompiled header, save space
set(CMAKE_DISABLE_PRECOMPILE_HEADERS ON)
# If support unity build, improve speed on headers
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set(CMAKE_UNITY_BUILD ON)
endif()
# If enable tests
option(BUILD_TESTING ON "If you want enable test, default: ON")
# If enable coverage
option(CODE_COVERAGE OFF "If you want enable coverage, default: OFF")
option(BUILD_DOCS_DOXYGEN ON "If you want build Doc with Doxygen, default: ON")
option(BUILD_ARDUINO ON "If you want build arduino lib, default: ON")

if (CMAKE_BUILD_TYPE)
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()


# Add Postfix for debug conf
set(CMAKE_DEBUG_POSTFIX "_d")

#set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
#set(CMAKE_C_COMPILER_LAUNCHER ccache)
#set(Var1 "This is variable 1" CACHE INTERNAL "")
#set(Var1 "This is variable 1" PARENT_SCOPE)

include(ConfigureCCache)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
#set(CMAKE_INSTALL_LIBDIR "lib")
#set(CMAKE_INSTALL_BINDIR "bin")
#set(CMAKE_INSTALL_INCLUDEDIR "include")
set(UNIT_TEST_BIN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/include")
#set(CTEST_BINARY_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

#set(AUTOGEN_BUILD_DIR "auto_gen")
#set(PROJECT_BINARY_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
#set(CTEST_BINARY_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
     OWNER_READ
     OWNER_WRITE
     OWNER_EXECUTE
     GROUP_READ
   )

#   ===================================================
#   ===============    Git submodule    ===============
#   ===================================================

#   ================    Git submodule    ================
include(ConfigureGit)

#   ================================================
#   ================    COMPILER    ================
#   ================================================
include(ConfigureCompiler)

#   ================    COMPILER DEF    ================
add_definitions(-DPROJECT_VERSION="${PROJECT_VERSION}")
add_definitions(-DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
add_definitions(-DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}")
add_definitions(-DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}")
add_definitions(-DCMAKE_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}")
add_definitions(-DCMAKE_C_COMPILER_ID="${CMAKE_C_COMPILER_ID}")
add_definitions(-DCMAKE_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}")
add_definitions(-DCMAKE_C_COMPILER_VERSION="${CMAKE_C_COMPILER_VERSION}")
add_definitions(-DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS}")


if(CMAKE_SIZEOF_VOID_P EQUAL 2)
    message(STATUS "16 bit mode")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(STATUS "32 bit mode")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "64 bit mode")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 16)
    message(STATUS "128 bit mode")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 32)
    message(STATUS "256 bit mode")
else()
    message(STATUS "unknow size")
endif()


#   =================================================
#   ==============    EXTERNAL LIBS    ==============
#   =================================================

include(ExternalProject)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

#   ================    Thread    ================
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

#   ================    OpenMP    ================
include(ConfigureOpenMP)

#   ================    SFML    ================
#set(SFML_ROOT "C:\\Project Files (x86)\\SFML")
include(ConfigureSFML)

#   ================    OGRE3D    ================
#include(ConfigureOGRE)
#setupOgre( OGRE_SOURCE, OGRE_BINARIES, OGRE_LIBRARIES, FALSE, FALSE )
#find_package(OGRE 2.1 REQUIRED)
#include_directories(${OGRE_INCLUDE_DIRS})

#${OGRE_LIBRARIES}


#   ================    Thread    ================


#   ================    BOOST    ================
include(ConfigureBoost)

#   ================    OPENCV    ================
include(ConfigureOpenCV)

#   ================    OPENGL    ================
include(ConfigureOpenGL)

#   ================    OPENCL    ================
include(ConfigureOpenCL)

#   ================    CUDA    ================
include(ConfigureCUDA)

#   ================    VULKAN    ================
include(ConfigureVulkan)

#   ================    OPENSSL    ================
include(ConfigureOpenSSL)

#   ================    Qt    ================
option(CMAKE_AUTOUIC ON)
option(CMAKE_AUTOMOC ON)
option(CMAKE_AUTORCC ON)
include(ConfigureQt)

#   ================    cppcheck    ================
#set(ENABLE_CODE_ANALYSIS "Run code analysis" OFF)

include(ConfigureCppCheck)

#   ================    Doxygen    ================
include(ConfigureDoxygen)

#   =================================================
#   =============    INTERNAL LIBS    ===============
#   =================================================
#set(BASEPATH "${CMAKE_SOURCE_DIR}")
#include_directories("${BASEPATH}")

set(lib_path "")

include_directories("${lib_path}lib/") #To avoid to use full path in headers
include_directories("${lib_path}lib/C/") #To avoid to use full path in headers


# C libs
add_subdirectory(${lib_path}lib/C/libhello "src/lib/C/libhello")
add_subdirectory(${lib_path}lib/C/string_c "src/lib/C/string_c")

add_subdirectory(${lib_path}lib/C/string_is "src/lib/C/string_is_c")
add_subdirectory(${lib_path}lib/C/pointer "src/lib/C/pointer_c")
add_subdirectory(${lib_path}lib/C/display "src/lib/C/display_c")
add_subdirectory(${lib_path}lib/C/integer "src/lib/C/integer_c")

# C++ libs
add_subdirectory(${lib_path}lib/convert_utils "src/lib/convert_utils")

add_subdirectory(${lib_path}lib/texture "src/lib/texture")

add_subdirectory(${lib_path}lib/math "src/lib/math")

add_subdirectory(${lib_path}lib/utils/opencv "src/lib/utils/opencv")

add_subdirectory(${lib_path}lib/utils/opengl "src/lib/utils/opengl")

#add_subdirectory(src/lib/utils/sfml "src/lib/utils/sfml")

add_subdirectory(${lib_path}lib/time "src/lib/time")

add_subdirectory(${lib_path}lib/cpu "src/lib/cpu")

add_subdirectory(${lib_path}lib/vector "src/lib/vector")

add_subdirectory(${lib_path}lib/thread "src/lib/thread")

add_subdirectory(${lib_path}lib/filesystem "src/lib/filesystem")

add_subdirectory(${lib_path}lib/string_lib "src/lib/string_lib")
if (OPENSSL_FOUND)
    add_subdirectory(${lib_path}lib/crypto "src/lib/crypto")
endif()
add_subdirectory(${lib_path}lib/std "src/lib/std")

if (CUDA_FOUND)
    add_subdirectory(${lib_path}lib/CUDA "src/lib/CUDA")
endif()

add_subdirectory(${lib_path}lib/python "src/lib/python")

if(BUILD_ARDUINO)
    #add_subdirectory(${lib_path}lib/arduino "src/lib/arduino")
    add_subdirectory(${lib_path}lib/ArduinoLinux "src/lib/ArduinoLinux")
endif(BUILD_ARDUINO)

#add_dependencies(hello ${EXTERNAL_LIB})



#   =================================================
#   =============    SOURCES FILES    ===============
#   =================================================

#   ================    FILES    ================
set(MAIN src/main.cpp)

set(SRCS
    )

set(HEADERS
    )

set(VIRUS 
    )

set(BENCH_ISPRIME ${lib_path}src/bench/bench_isprime.cpp)

set(BENCH_VECTOR_COPY ${lib_path}src/bench/bench_vector_copy.cpp)

set(BENCH_VECTOR_SORT ${lib_path}src/bench/bench_vector_sort.cpp)

set(BENCH_DIGITS_COUNT ${lib_path}src/bench/bench_digits_count.cpp)

set(BENCH_FIND_MAX ${lib_path}src/bench/bench_find_max.cpp)

set(BENCH_PC ${lib_path}src/bench/benchmark_pc.cpp)

set(BENCH_CRC32 ${lib_path}src/bench/bench_crc32.cpp)

set(LIST_ALL_FILE ${lib_path}src/exemple/list_all_file.cpp)

set(PARSE_CSV ${lib_path}src/exemple/parse_csv.cpp)

set(GTA_SA_CHEAT ${lib_path}src/exemple/GTA_SA_cheat.cpp)

if (OpenCL_FOUND)
    set(OPENCL_TEST ${lib_path}src/exemple/OpenCL/OpenCL_test.cpp)
    set(OPENCL_TEST2 ${lib_path}src/exemple/OpenCL/OpenCL_test2.cpp)
    set(OPENCL_CRC32 ${lib_path}src/exemple/OpenCL/OpenCL_crc32.cpp)
    set(OPENCL_FINDSTRINGINV ${lib_path}src/exemple/OpenCL/OpenCL_findStringInv.cpp)
    set(OPENCL_GTA_SA_GPU ${lib_path}src/exemple/OpenCL/OpenCL_GTA_SA_cheat_GPU.cpp)
endif()

if (OPENSSL_FOUND)
    set(LIST_ALL_FILE_HASH ${lib_path}src/exemple/list_all_file_hash.cpp)
    set(CRYPTO_AES ${lib_path}src/exemple/crypto_AES256.cpp)
    set(CRYPTO_RSA ${lib_path}src/exemple/crypto_RSA.cpp)
endif()

set(COUNT_ALL_FILE ${lib_path}src/exemple/count_all_file.cpp)

if (OPENCV_FOUND)
    set(IMAGE_DIFF ${lib_path}src/exemple/image_diff.cpp)
    set(LIST_ALL_FILE_OPENCV ${lib_path}src/exemple/list_all_file_opencv.cpp)
endif()

#   ====================================================
#   ====================    LIBS    ====================
#   ====================================================

#set(MAIN_LLIBS Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Quick Qt${QT_VERSION_MAJOR}::Multimedia sfml-graphics sfml-window sfml-audio sfml-network sfml-system ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${OpenCV_LIBS})
#set(MAIN_LLIBS sfml-graphics sfml-window sfml-audio sfml-network sfml-system ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${OpenCV_LIBS})

set(MAIN_LLIBS "")

if(Boost_FOUND)
    set(MAIN_LLIBS ${MAIN_LLIBS} ${Boost_LIBRARIES})
endif()

if(SFML_FOUND)
    set(MAIN_LLIBS ${MAIN_LLIBS} sfml-graphics sfml-window sfml-audio sfml-network sfml-system)
endif()

if (OPENCV_FOUND)
    set(MAIN_LLIBS ${MAIN_LLIBS} ${OpenCV_LIBS})
endif()

if (OPENGL_FOUND)
    set(MAIN_LLIBS ${MAIN_LLIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${CUDA_LIBRARY})
endif()

if (QT_FOUND)
    set(MAIN_LLIBS ${MAIN_LLIBS} Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Quick Qt${QT_VERSION_MAJOR}::Multimedia)
endif()

#   ==================================================
#   ===============    EXEC FILES    =================
#   ==================================================

add_executable(bench_isprime ${BENCH_ISPRIME})
target_link_libraries(bench_isprime PRIVATE thread Threads::Threads vector math time)
add_dependencies(bench_isprime vector math time thread)
set_target_properties(bench_isprime PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

add_executable(bench_vector_copy ${BENCH_VECTOR_COPY})
target_link_libraries(bench_vector_copy PRIVATE thread Threads::Threads vector math time)
add_dependencies(bench_vector_copy thread vector math time)
set_target_properties(bench_vector_copy PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

add_executable(bench_vector_sort ${BENCH_VECTOR_SORT})
target_link_libraries(bench_vector_sort PRIVATE thread Threads::Threads vector math time)
add_dependencies(bench_vector_sort thread vector math time)
set_target_properties(bench_vector_sort PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

add_executable(bench_digits_count ${BENCH_DIGITS_COUNT})
target_link_libraries(bench_digits_count PRIVATE thread Threads::Threads vector math time)
add_dependencies(bench_digits_count thread vector math time)
set_target_properties(bench_digits_count PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

add_executable(bench_find_max ${BENCH_FIND_MAX})
target_link_libraries(bench_find_max PRIVATE thread Threads::Threads vector math time)
add_dependencies(bench_find_max thread vector math time)
set_target_properties(bench_find_max PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

add_executable(bench_pc ${BENCH_PC})
target_link_libraries(bench_pc PRIVATE time vector)
add_dependencies(bench_pc time vector)
set_target_properties(bench_pc PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

add_executable(bench_crc32 ${BENCH_CRC32})
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_crc32 PRIVATE thread Threads::Threads vector my_crypto math time OpenMP::OpenMP_CXX)
else()
    target_link_libraries(bench_crc32 PRIVATE thread Threads::Threads vector my_crypto math time)
endif()
add_dependencies(bench_crc32 thread Threads::Threads vector my_crypto math time)
set_target_properties(bench_crc32 PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)


if(Boost_FOUND)
    add_executable(gta_sa_cheat ${GTA_SA_CHEAT})
    if(OpenMP_FOUND)
        target_link_libraries(gta_sa_cheat PRIVATE string_lib time OpenMP::OpenMP_CXX)
        add_dependencies(gta_sa_cheat string_lib time OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(gta_sa_cheat PRIVATE string_lib time)
        add_dependencies(gta_sa_cheat string_lib time)
    endif()
    set_target_properties(gta_sa_cheat PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

    add_executable(list_all_file ${LIST_ALL_FILE})
    target_link_libraries(list_all_file PRIVATE ${Boost_LIBRARIES} filesystem)
    set_target_properties(list_all_file PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
endif()

if(OpenCL_FOUND)
    add_executable(opencl_test ${OPENCL_TEST})
    target_link_libraries(opencl_test PRIVATE ${OpenCL_LIBRARY})
    set_target_properties(opencl_test PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

    add_executable(opencl_test2 ${OPENCL_TEST2})
    target_link_libraries(opencl_test2 PRIVATE ${OpenCL_LIBRARY})
    set_target_properties(opencl_test2 PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

    add_executable(opencl_findstringinv ${OPENCL_FINDSTRINGINV})
    target_link_libraries(opencl_findstringinv PRIVATE ${OpenCL_LIBRARY} string_lib)
    set_target_properties(opencl_findstringinv PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

    add_executable(opencl_gta_sa_gpu ${OPENCL_GTA_SA_GPU})
    target_link_libraries(opencl_gta_sa_gpu PRIVATE ${OpenCL_LIBRARY} string_lib)
    set_target_properties(opencl_gta_sa_gpu PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

    if (OPENSSL_FOUND)
        add_executable(opencl_crc32 ${OPENCL_CRC32})
        target_link_libraries(opencl_crc32 PRIVATE ${OpenCL_LIBRARY} my_crypto time)
        set_target_properties(opencl_crc32 PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    endif()
endif()

add_executable(parse_csv ${PARSE_CSV})
target_link_libraries(parse_csv PRIVATE string_lib)
add_dependencies(parse_csv string_lib)
set_target_properties(parse_csv PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

if (OPENSSL_FOUND)
    add_executable(list_all_file_hash ${LIST_ALL_FILE_HASH})
    if(OpenMP_FOUND)
        target_link_libraries(list_all_file_hash PRIVATE ${Boost_LIBRARIES} ${OPENSSL_LIBRARIES} my_crypto OpenMP::OpenMP_CXX time)
        add_dependencies(list_all_file_hash my_crypto OpenMP::OpenMP_CXX time)
    else()
        target_link_libraries(list_all_file_hash PRIVATE ${Boost_LIBRARIES} ${OPENSSL_LIBRARIES} my_crypto time)
        add_dependencies(list_all_file_hash my_crypto time)
    endif()
    set_target_properties(list_all_file_hash PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

    add_executable(crypto_AES ${CRYPTO_AES})
    target_link_libraries(crypto_AES PRIVATE ${Boost_LIBRARIES} ${OPENSSL_LIBRARIES} filesystem my_crypto thread time)
    add_dependencies(crypto_AES filesystem thread time)
    set_target_properties(crypto_AES PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

    add_executable(crypto_RSA ${CRYPTO_RSA})
    target_link_libraries(crypto_RSA PRIVATE ${Boost_LIBRARIES} ${OPENSSL_LIBRARIES} filesystem my_crypto thread time)
    add_dependencies(crypto_RSA my_crypto thread time filesystem)
    set_target_properties(crypto_RSA PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
endif()

if (OPENCV_FOUND)
    add_executable(list_all_file_opencv ${LIST_ALL_FILE_OPENCV})
    target_link_libraries(list_all_file_opencv PRIVATE filesystem ${OpenCV_LIBS})
    add_dependencies(list_all_file_opencv filesystem ${OpenCV_LIBS})
    set_target_properties(list_all_file_opencv PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)

    add_executable(image_diff ${IMAGE_DIFF})
    target_link_libraries(image_diff PRIVATE filesystem ${OpenCV_LIBS} utils_opencv)
    set_target_properties(image_diff PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
endif()

#   ================================================
#   ================    TESTING     ================
#   ================================================
set(ENABLE_TESTS "Enable tests" ON)

if(Boost_FOUND AND ENABLE_TESTS)
    enable_testing()
    #Test getSchwarzschild
    add_executable(test_getSchwarzschild ${lib_path}src/test/math/test_getSchwarzschild.cpp)
    target_link_libraries(test_getSchwarzschild PRIVATE ${Boost_LIBRARIES} math)
    set_target_properties(test_getSchwarzschild PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    add_test(NAME test_getSchwarzschild COMMAND ${CMAKE_COMMAND} --build . --target test_getSchwarzschild)

    #Test getGravitationalAttraction
    add_executable(test_getGravitationalAttraction ${lib_path}src/test/math/test_getGravitationalAttraction.cpp)
    target_link_libraries(test_getGravitationalAttraction PRIVATE ${Boost_LIBRARIES} math)
    set_target_properties(test_getGravitationalAttraction PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    add_test(NAME test_getGravitationalAttraction COMMAND test_getGravitationalAttraction)

    #Test sphere
    add_executable(test_sphere ${lib_path}src/test/math/test_sphere.cpp)
    target_link_libraries(test_sphere PRIVATE ${Boost_LIBRARIES} quadmath math)
    set_target_properties(test_sphere PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    add_test(NAME test_sphere COMMAND test_sphere)

    #Test cylinder
    add_executable(test_cylinder ${lib_path}src/test/math/test_cylinder.cpp)
    target_link_libraries(test_cylinder PRIVATE ${Boost_LIBRARIES} math)
    set_target_properties(test_cylinder PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    add_test(NAME test_cylinder COMMAND test_cylinder)

    #Test vector sort
    add_executable(test_vector_sort ${lib_path}src/test/vector/test_vector_sort.cpp)
    target_link_libraries(test_vector_sort PRIVATE ${Boost_LIBRARIES} vector)
    set_target_properties(test_vector_sort PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    add_test(NAME test_vector_sort COMMAND test_vector_sort)

    #Test vector copy
    add_executable(test_vector_copy ${lib_path}src/test/vector/test_vector_copy.cpp)
    target_link_libraries(test_vector_copy PRIVATE ${Boost_LIBRARIES} vector)
    set_target_properties(test_vector_copy PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    add_test(NAME test_vector_copy COMMAND test_vector_copy)

    #Test vector copy
    add_executable(test_vector_max_simd ${lib_path}src/test/vector/test_vector_max_simd.cpp)
    target_link_libraries(test_vector_max_simd PRIVATE ${Boost_LIBRARIES} vector)
    set_target_properties(test_vector_max_simd PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    add_test(NAME test_vector_max_simd COMMAND test_vector_max_simd)

    add_executable(test_string_seq ${lib_path}src/test/string/test_string_seq.cpp)
    target_link_libraries(test_string_seq PRIVATE ${Boost_LIBRARIES} string_lib)
    set_target_properties(test_string_seq PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    add_test(NAME test_string_seq COMMAND test_string_seq)
endif()

if (OPENSSL_FOUND AND Boost_FOUND AND ENABLE_TESTS)
    #Test crypto
    add_executable(test_crypto ${lib_path}src/test/crypto/test_crypto.cpp)
    target_link_libraries(test_crypto PRIVATE ${Boost_LIBRARIES} ${OPENSSL_LIBRARIES} my_crypto)
    set_target_properties(test_crypto PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    add_test(NAME test_crypto COMMAND test_crypto)

    #Test crypto AES
    add_executable(test_crypto_aes ${lib_path}src/test/crypto/test_crypto_AES.cpp)
    target_link_libraries(test_crypto_aes PRIVATE ${Boost_LIBRARIES} ${OPENSSL_LIBRARIES} my_crypto)
    set_target_properties(test_crypto_aes PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    add_test(NAME test_crypto_aes COMMAND test_crypto_aes)

    add_executable(test_crypto_CRC ${lib_path}src/test/crypto/test_crypto_CRC.cpp)
    target_link_libraries(test_crypto_CRC PRIVATE ${Boost_LIBRARIES} my_crypto)
    set_target_properties(test_crypto_CRC PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    add_test(NAME test_crypto_CRC COMMAND test_crypto_CRC)
endif()

#Test C string
if(Boost_FOUND AND ENABLE_TESTS)
    
    set(TEST_REV_STR test_rev_str)
    add_executable(${TEST_REV_STR} ${lib_path}src/test/C/string_c/test_string.cpp)
    target_link_libraries(${TEST_REV_STR} PRIVATE ${Boost_LIBRARIES} string_c)
    set_target_properties(${TEST_REV_STR} PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    add_test(NAME ${TEST_REV_STR} COMMAND ${TEST_REV_STR})

    set(TEST_IS_NUM test_my_isnum)
    add_executable(${TEST_IS_NUM} ${lib_path}src/test/C/string_is/test_my_isnum.cpp)
    target_link_libraries(${TEST_IS_NUM} PRIVATE ${Boost_LIBRARIES} string_is_c)
    set_target_properties(${TEST_IS_NUM} PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    add_test(NAME ${TEST_IS_NUM} COMMAND ${TEST_IS_NUM})

    set(TEST_STR_IS_NUM test_my_str_isnum)
    add_executable(${TEST_STR_IS_NUM} ${lib_path}src/test/C/string_is/test_my_str_isnum.cpp)
    target_link_libraries(${TEST_STR_IS_NUM} PRIVATE ${Boost_LIBRARIES} string_is_c)
    set_target_properties(${TEST_STR_IS_NUM} PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    add_test(NAME ${TEST_STR_IS_NUM} COMMAND ${TEST_STR_IS_NUM})
    
    #Test power
    set(TEST_POWER test_power)
    add_executable(${TEST_POWER} ${lib_path}src/test/math/test_power.cpp)
    target_link_libraries(${TEST_POWER} PRIVATE ${Boost_LIBRARIES} math)
    set_target_properties(${TEST_POWER} PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
    add_test(NAME ${TEST_POWER} COMMAND ${TEST_POWER})
endif()
#add_test(NAME ${TEST_POWER} COMMAND ${CMAKE_COMMAND} --build . --target ${TEST_POWER})

#add_custom_command(
#     TARGET test_power
#     COMMENT "Run tests"
#     POST_BUILD 
#     WORKING_DIRECTORY .
#     COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failures --extra-verbose
#)

#   ================================================
#   ===============    COPY FILES    ===============
#   ================================================

#file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/screenshot)

# For OpenCL_test.cpp
configure_file(${lib_path}lib/opencl/kernels/vector_add_kernel.cl ${CMAKE_BINARY_DIR}/kernels/vector_add_kernel.cl COPYONLY)
# For OpenCL_test2.cpp
configure_file(${lib_path}lib/opencl/kernels/simple_add.cl ${CMAKE_BINARY_DIR}/kernels/simple_add.cl COPYONLY)
# For OpenCL_crc32.cpp
configure_file(${lib_path}lib/opencl/kernels/crc_kernel.cl ${CMAKE_BINARY_DIR}/kernels/crc_kernel.cl COPYONLY)
# For OpenCL_findStringInv.cpp
configure_file(${lib_path}lib/opencl/kernels/findStringInv.cl ${CMAKE_BINARY_DIR}/kernels/findStringInv.cl COPYONLY)
# For GTA SA Alternative cheats
configure_file(${lib_path}lib/opencl/kernels/gta_sa.cl ${CMAKE_BINARY_DIR}/kernels/gta_sa.cl COPYONLY)

#add_custom_command(
#    TARGET game
#    COMMENT "CP Texture"
#    POST_BUILD 
#    WORKING_DIRECTORY .
#    COMMAND cp -R -u -p "../texture/PIPOYA FREE RPG Character Sprites 32x32" "texture/PIPOYA FREE RPG Character Sprites 32x32"
#)

#   =================================================
#   ===============    UNSET VALUE    ===============
#   =================================================


#unset(CMAKE_BUILD_TYPE CACHE)

#   ================================================
#   =================    CPack    ==================
#   ================================================
#https://stackoverflow.com/questions/32460371/adding-files-to-archive-packages-only-in-cpack
include(InstallRequiredSystemLibraries)

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "My funky project")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_VENDOR "Ben")
set(CPACK_PACKAGE_NAME "BenLib")
set(CPACK_PACKAGE_CONTACT "bensuperpc@gmail.com")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/Bensuperpc/BenLib")

#set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
#set(CPACK_COMPONENTS_ALL_IN_ONE_PACKAGE ON)

set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_WELCOME "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

set(CPACK_COMPONENT_APPLICATIONS_DISPLAY_NAME "MyLib Application")
set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "Libraries")
set(CPACK_COMPONENT_HEADERS_DISPLAY_NAME "C/C++ Headers")

set(CPACK_COMPONENT_APPLICATIONS_DESCRIPTION
  "An useful applications")
set(CPACK_COMPONENT_LIBRARIES_DESCRIPTION
  "Shared libraries")
set(CPACK_COMPONENT_HEADERS_DESCRIPTION
  "C/C++ header files for use with BenLib")

set(CPACK_COMPONENT_APPLICATIONS_GROUP "Runtime")
set(CPACK_COMPONENT_LIBRARIES_GROUP "Development")
set(CPACK_COMPONENT_HEADERS_GROUP "Development")  

set(CPACK_COMPONENT_GROUP_DEVELOPMENT_DESCRIPTION
  "All of the tools you'll need to develop")


#set(CPACK_COMPONENT_HEADERS_DEPENDS libraries)

set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")

set(CPACK_PRE_BUILD_SCRIPTS "")
set(CPACK_POST_BUILD_SCRIPTS "")

set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package")
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set(CPACK_PACKAGE_INSTALL_DIRECTORY "CMake ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")

if(WIN32 AND NOT UNIX)
  set(CPACK_PACKAGE_ICON "${CMake_SOURCE_DIR}/Utilities/Release\\\\InstallIcon.bmp")
  set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\MyExecutable.exe")
  set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} My Famous Project")
  set(CPACK_NSIS_HELP_LINK "http:\\\\\\\\www.bensuperpc.org")
  set(CPACK_NSIS_URL_INFO_ABOUT "http:\\\\\\\\www.bensuperpc.com")
  set(CPACK_NSIS_CONTACT "${CPACK_PACKAGE_CONTACT}")
  set(CPACK_NSIS_MODIFY_PATH ON)
else()
  set(CPACK_STRIP_FILES "bin/MyExecutable")
  set(CPACK_SOURCE_STRIP_FILES "")
endif()

set(CPACK_PACKAGE_RELEASE "${CMAKE_BUILD_TYPE}")
set(CPACK_PACKAGE_EXECUTABLES "MyExecutable" "My Executable")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")

set(CPACK_PROJECT_CONFIG_FILE ${CMAKE_SOURCE_DIR}/package.txt)
#set(CPACK_VERBATIM_VARIABLES true)
set(CPACK_COMPONENTS_ALL applications libraries headers tests)
set(CPACK_GENERATOR "External;TZ;STGZ;TGZ;TZST;TBZ2;RPM;ZIP;7Z;DEB")
set(CPACK_PACKAGE_CHECKSUM "SHA3_512")
set(CPACK_CMAKE_GENERATOR Ninja)

set(CPACK_SOURCE_IGNORE_FILE "")

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20.0")
    set(CPACK_THREADS 0)
endif()

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
    set(CPACK_ARCHIVE_THREADS 16)
endif()

# ====== RPM ======
set(CPACK_RPM_PACKAGE_GROUP "Bensuperpc")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_PACKAGE_RELOCATABLE ON)
set(CPACK_RPM_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}")
set(CPACK_RPM_PACKAGE_RELEASE ON)
set(CPACK_RPM_PACKAGE_DESCRIPTION "${CPACK_PACKAGE_DESCRIPTION_SUMMARY}")
set(CPACK_RPM_PACKAGE_AUTOREQ ON)

set(CPACK_RPM_PACKAGE_REQUIRES "clang >= 10.0.0, cmake >= 3.0")
set(CPACK_RPM_PACKAGE_SUGGESTS "")
set(CPACK_RPM_PACKAGE_CONFLICTS "libxml2")

#set(CPACK_RPM_PACKAGE_SUGGESTS "clang >= 10.0.0, cmake >= 3.0")
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/home")
list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/home/bensuperpc")
#set(CPACK_RPM_COMPRESSION_TYPE "xz")
set(CPACK_RPM_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.rpm")
set (CPACK_RPM_POST_INSTALL_SCRIPT_FILE "")
set (CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "")



# ====== DEB ======
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Bensuperpc")
set(CPACK_DEBIAN_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")

set(CPACK_DEBIAN_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "clang (>= 10.0.0), libc6 (>= 3.0)")
set(CPACK_DEBIAN_PACKAGE_BREAKS "")
set(CPACK_DEBIAN_PACKAGE_CONFLICTS "")
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "")
set(CPACK_DEBIAN_PACKAGE_SECTION "dev")
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64") #or i386

set(CPACK_DEBIAN_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.deb")
set(CPACK_DEB_COMPONENT_INSTALL OFF)
set(CPACK_DEBIAN_COMPRESSION_TYPE "lzma")
set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION FALSE)

# ====== DIRECTORY ======
install(DIRECTORY ${CMAKE_SOURCE_DIR}/lib/opencl/kernels/
        DESTINATION kernels
        COMPONENT ResourcesCursors)

# ====== FILES ======
install(FILES
  ${CPACK_RESOURCE_FILE_README}
  ${CPACK_RESOURCE_FILE_LICENSE}
  DESTINATION Metadata
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/tools/graph.py
  DESTINATION tools)

# ====== LIBS ======
#install(TARGETS my_crypto
#  ARCHIVE
#  DESTINATION lib
#  COMPONENT libraries)

# ====== BIN ======
install(TARGETS list_all_file_hash
  RUNTIME
  DESTINATION bin
  COMPONENT applications)
install(TARGETS gta_sa_cheat
  RUNTIME
  DESTINATION bin
  COMPONENT applications)
install(TARGETS bench_isprime
  RUNTIME
  DESTINATION bin
  COMPONENT applications)
install(TARGETS bench_vector_copy
  RUNTIME
  DESTINATION bin
  COMPONENT applications)
install(TARGETS bench_vector_sort
  RUNTIME
  DESTINATION bin
  COMPONENT applications)
install(TARGETS bench_digits_count
  RUNTIME
  DESTINATION bin
  COMPONENT applications)
install(TARGETS bench_find_max
  RUNTIME
  DESTINATION bin
  COMPONENT applications)
install(TARGETS bench_pc
  RUNTIME
  DESTINATION bin
  COMPONENT applications)
install(TARGETS bench_crc32
  RUNTIME
  DESTINATION bin
  COMPONENT applications)
if(OpenCL_FOUND)
    install(TARGETS opencl_test
        RUNTIME
        DESTINATION bin
        COMPONENT applications)
    install(TARGETS opencl_test2
        RUNTIME
        DESTINATION bin
        COMPONENT applications)
    install(TARGETS opencl_findstringinv
        RUNTIME
        DESTINATION bin
        COMPONENT applications)
    install(TARGETS opencl_gta_sa_gpu
        RUNTIME
        DESTINATION bin
        COMPONENT applications)
    if (OPENSSL_FOUND)
        install(TARGETS opencl_crc32
            RUNTIME
            DESTINATION bin
            COMPONENT applications)
  endif()
endif()
# ====== TESTS ======
if(ENABLE_TESTS)
    install(TARGETS test_power
        RUNTIME
        DESTINATION tests
        COMPONENT tests)
    install(TARGETS test_my_str_isnum
        RUNTIME
        DESTINATION tests
        COMPONENT tests)
    install(TARGETS test_my_isnum
        RUNTIME
        DESTINATION tests
        COMPONENT tests)
    install(TARGETS test_rev_str
        RUNTIME
        DESTINATION tests
        COMPONENT tests)
    install(TARGETS test_vector_copy
        RUNTIME
        DESTINATION tests
        COMPONENT tests)
    install(TARGETS test_vector_max_simd
        RUNTIME
        DESTINATION tests
        COMPONENT tests)
    install(TARGETS test_string_seq
        RUNTIME
        DESTINATION tests
        COMPONENT tests)
    install(TARGETS test_vector_sort
        RUNTIME
        DESTINATION tests
        COMPONENT tests)
    install(TARGETS test_cylinder
        RUNTIME
        DESTINATION tests
        COMPONENT tests)
    install(TARGETS test_sphere
        RUNTIME
        DESTINATION tests
        COMPONENT tests)
    install(TARGETS test_getGravitationalAttraction
        RUNTIME
        DESTINATION tests
        COMPONENT tests)
    install(TARGETS test_getSchwarzschild
        RUNTIME
        DESTINATION tests
        COMPONENT tests)

    if (OPENSSL_FOUND)
        install(TARGETS test_crypto_CRC
            RUNTIME
            DESTINATION tests
            COMPONENT tests)
        install(TARGETS test_crypto_aes
            RUNTIME
            DESTINATION tests
            COMPONENT tests)
        install(TARGETS test_crypto
            RUNTIME
            DESTINATION tests
            COMPONENT tests)
    endif()

endif()

#install(FILES mylib.h
#  DESTINATION include
#  COMPONENT headers)

# Section CPack
include (CPack)
