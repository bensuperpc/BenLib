cmake_minimum_required (VERSION 3.13)
project(RPG VERSION 0.0.1 LANGUAGES CXX C)

#   ==================================================
#   ================    CMAKE CONF    ================
#   ==================================================


set(CMAKE_INCLUDE_CURRENT_DIR 1)
set(CMAKE_VERBOSE_MAKEFILE 1)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    set(CMAKE_UNITY_BUILD 1)
    message(STATUS "CMAKE_UNITY_BUILD : 1")
else()
    set(CMAKE_UNITY_BUILD 0)
    message(STATUS "CMAKE_UNITY_BUILD : 0")
endif()

#set(Var1 "This is variable 1" CACHE INTERNAL "")
#set(Var1 "This is variable 1" PARENT_SCOPE)

#   ================================================
#   ================    COMPILER    ================
#   ================================================

if(NOT CMAKE_C_COMPILER)
#    set(CMAKE_C_COMPILER /usr/bin/clang)
    set(CMAKE_C_COMPILER /usr/bin/gcc)
endif()
if(NOT CMAKE_CXX_COMPILER)
#    set(CMAKE_CXX_COMPILER /usr/bin/clang++)
    set(CMAKE_CXX_COMPILER /usr/bin/g++)
endif()

#=== C++ VERSION CHECK ===
if(NOT CMAKE_CXX_STANDARD)
    include(CheckCXXCompilerFlag)

    CHECK_CXX_COMPILER_FLAG("-std=c++20" COMPILER_SUPPORTS_CXX20)
    CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
    CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
    CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

    if(COMPILER_SUPPORTS_CXX20)
        set(CMAKE_CXX_STANDARD 20)
        message(STATUS "C++20: OK")
    elseif(COMPILER_SUPPORTS_CXX17)
        set(CMAKE_CXX_STANDARD 17)
        message(STATUS "C++17: OK")
    elseif(COMPILER_SUPPORTS_CXX14)
        set(CMAKE_CXX_STANDARD 14)
        message(STATUS "C++14: OK")
    elseif(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_STANDARD 11)
        message(STATUS "C++11: OK")
        message(WARNING "C++11 is old, please use newer compiler.")
    else()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 or above support. Please use a different C++ compiler.")
    endif()

    # Compiler-specific C++11 activation.
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        # require at least gcc 4.8
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
            message(FATAL_ERROR "GCC version must be at least 4.8!")
            set(CMAKE_CXX_STANDARD 14)
        endif()
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        # require at least clang 3.2
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
            message(FATAL_ERROR "Clang version must be at least 3.2!")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0")
        message(FATAL_ERROR "Insufficient msvc version")
        endif()
    else()
        message(WARNING "You are using an unsupported compiler! Compilation has only been tested with Clang and GCC.")
    endif()
endif()


set(CMAKE_CXX_STANDARD_REQUIRED 1)
if(NOT CMAKE_CXX_EXTENSIONS)
    set(CMAKE_CXX_EXTENSIONS 1)
endif()

#=== OPTIMIZATION FLAGS ===
#-mavx2 -flto
set(OPTI_COMPILER_REL "-march=native -s -ffunction-sections -fdata-sections -Wl,--gc-sections -Wl,--strip-all")

#=== SECURITY RELEASE FLAGS ===

set(SEC_COMPILER_REL "-fstack-protector-all -Werror=format-security -fstack-protector-strong -fexceptions -D_FORTIFY_SOURCE=2 -fPIE")
#For Ubuntu 18.04 and 16.04
CHECK_CXX_COMPILER_FLAG("-fstack-clash-protection" COMPILER_SUPPORTS_SEC_FLAG)
if(NOT COMPILER_SUPPORTS_SEC_FLAG)
    list(APPEND CHECK_CXX_COMPILER_FLAG "/usr/share/SFML/cmake/Modules")
endif()

set(SEC_COMPILER_LINK_REL "-fPIC -Wl,-z,now -Wl,-z,relro")


set(WARNINGS_COMPILER_C "-Wall -Wpedantic -Wextra -Wstrict-prototypes -Wmissing-prototypes -Wfloat-equal -Wundef -Wshadow -Wpointer-arith -Wstrict-overflow=5 -Wswitch-default -Wunreachable-code -Wcast-align")
set(CMAKE_C_FLAGS                "${WARNINGS_COMPILER_C} -std=c99 -pipe")
set(CMAKE_C_FLAGS_RELEASE        "-O3 ${OPTI_COMPILER_REL} ${SEC_COMPILER_REL}")
set(CMAKE_C_FLAGS_MINSIZEREL     "-Os ${OPTI_COMPILER_REL} ${SEC_COMPILER_REL}")
set(CMAKE_C_FLAGS_DEBUG          "-g3 -Og -ggdb3 -v")
set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_C_FLAGS_DEBUG} --coverage")

#-Wold-style-cast
set(WARNINGS_COMPILER_CXX "-Wpedantic -Wall -Wextra -Wshadow -Wstrict-overflow=5 -Wmissing-declarations -Wundef -Wstack-protector -Wno-unused-parameter -Woverloaded-virtual  -Wdisabled-optimization -Winline -Wredundant-decls -Wsign-conversion -Wformat-nonliteral -Wformat-security -Wmissing-include-dirs -Wmissing-noreturn -Wimport -Winit-self -Winvalid-pch -Wstrict-aliasing=2 -Wswitch-default -Wunreachable-code -Wunused -Wwrite-strings -Wcast-align -Wcast-qual -Wfloat-equal -Wvariadic-macros -Wpacked  -Wpadded -Wpointer-arith -Weffc++ -Wformat=2 -Wfloat-equal -Wabi=11")
#-Wdouble-promotion -fstack-usage
set(WARNINGS_COMPILER_CXX "${WARNINGS_COMPILER_CXX} -Wduplicated-branches -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wnull-dereference -Wuseless-cast")

# Removed : -Wconversion
#-ftime-report -static -lrt -pthread -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -fprofile-arcs -ftest-coverage

#Add -lstdc++fs for old compiler
set(CMAKE_CXX_FLAGS                "${WARNINGS_COMPILER_CXX} -pipe -pthread -lstdc++fs -Wl,-rpath,../lib -Wl,-rpath,../external/lib -pthread")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os ${OPTI_COMPILER_REL} ${SEC_COMPILER_REL} -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 ${OPTI_COMPILER_REL} ${SEC_COMPILER_REL} -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG          "-g3 -Og -ggdb3 -v")
set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} --coverage")

set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${SEC_COMPILER_LINK_REL}")
set(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "${SEC_COMPILER_LINK_REL}")
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "${CMAKE_EXE_LINKER_FLAGS_DEBUG} --coverage")

set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} --coverage")

#set(BUILD_SHARED_LIBRARIES 0)
#set(CMAKE_EXE_LINKER_FLAGS "-static")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(PROJECT_BINARY_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
#set(CTEST_BINARY_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
#   ================    COMPILER DEF    ================


add_definitions(-DPROJECT_VERSION="${PROJECT_VERSION}")
add_definitions(-DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS})
add_definitions(-DCMAKE_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}")
add_definitions(-DCMAKE_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}")

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(STATUS "32 bit mode")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "64 bit mode")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 16)
    message(STATUS "128 bit mode")
else()
    message(STATUS "unknow size")
endif()

#   =================================================
#   ==============    EXTERNAL LIBS    ==============
#   =================================================

include(ExternalProject)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

#ExternalProject_Add(opencv_lib
#    GIT_REPOSITORY https://github.com/boostorg/boost
#    GIT_TAG 4.4.0
#    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    #CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
#)



#   ================    SFML    ================
#set(SFML_USE_STATIC_STD_LIBS 0)
#set(SFML_STATIC_LIBRARIES 0)

set(SFML_ROOT "C:\\Project Files (x86)\\SFML")

find_package(SFML 2.5 COMPONENTS graphics audio network system QUIET)

if(SFML_FOUND)
    message(STATUS "SFML: FOUND")
    add_compile_definitions(SFML_VERSION="${SFML_VERSION}")
else()
    message(STATUS "SFML: NOT FOUND")
    if(NOT EXTERNAL_LIB)
        set(EXTERNAL_LIB "")
    endif()
    string(APPEND EXTERNAL_LIB "sfml_lib")
    ExternalProject_Add(sfml_lib
        GIT_REPOSITORY https://github.com/SFML/SFML.git
        GIT_TAG 2.5.1
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
        #CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
    )
endif()

#   ================    BOOST    ================
#set( Boost_USE_STATIC_LIBS 0 )
#set( Boost_USE_MULTITHREADED 1 )
#set( Boost_USE_STATIC_RUNTIME 0 )

#find_package( Boost 1.72.0 COMPONENTS thread system fiber context program_options filesystem REQUIRED)
find_package(Boost 1.67.0 COMPONENTS thread filesystem system unit_test_framework QUIET)

if(Boost_FOUND)
    message(STATUS "Boost FOUND")
    include_directories( ${Boost_INCLUDE_DIRS} )
    link_directories(${Boost_LIBRARIES})
else()
    message(STATUS "Boost: NOT FOUND")
endif()

#   ================    OPENCV    ================
find_package(OpenCV REQUIRED QUIET)

if (OPENCV_FOUND)
    INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS})
    message(STATUS "OPENCV: FOUND")
else()
    message(STATUS "OPENCV: NOT FOUND")
    if(NOT EXTERNAL_LIB)
        set(EXTERNAL_LIB "")
    endif()
    string(APPEND EXTERNAL_LIB "opencv_lib")
    ExternalProject_Add(opencv_lib
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG 4.4.0
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
        #CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
    )
endif()


#   ================    OPENGL    ================
find_package(OpenGL REQUIRED QUIET)
find_package(GLUT REQUIRED QUIET)

if (OPENGL_FOUND)
    include_directories( ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} )
    link_directories(${OPENGL_LIBRARIES})
    message(STATUS "OPENCL: FOUND")
else()
    message(STATUS "OPENCL: NOT FOUND")
endif()

#   ================    VULKAN    ================
find_package(Vulkan QUIET)

if (VULKAN_FOUND)
    include_directories( ${Vulkan_INCLUDE_DIR})
    link_directories(${Vulkan_LIBRARY})
    message(STATUS "VULKAN: FOUND")
else()
    message(STATUS "VULKAN: NOT FOUND")
endif()

#   ================    Qt    ================

#set(CMAKE_AUTOUIC ON)
#set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTORCC ON)

#find_package(Qt5 COMPONENTS Quick Widgets Core Gui Multimedia Network REQUIRED)
#find_package(Qt5QuickCompiler)
#qtquick_compiler_add_resources(RESOURCES example.qrc)
#qt5_use_modules(myapp Quick Widgets Core Gui Multimedia Network)

#find_package(QT NAMES Qt6 Qt5 COMPONENTS Core Quick Multimedia REQUIRED)
#find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core Quick Multimedia REQUIRED)

if (QT_FOUND)
    message(STATUS "QT: FOUND")
    #include_directories( ${QT_INCLUDES})
else()
    message(STATUS "QT: NOT FOUND")
endif()

#   ================    DOCTEST    ================
find_package(doctest QUIET)

if (DOCTEST_FOUND)
    include_directories( ${doctest_INCLUDE_DIR})
    link_directories(${doctest_LIBRARY})
    message(STATUS "DOCTEST: FOUND")
else()
    message(STATUS "DOCTEST: NOT FOUND")
endif()

if(EXTERNAL_LIB)
    include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
    link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)
endif()

#   =================================================
#   =============    INTERNAL LIBS    ===============
#   =================================================

# C libs
add_subdirectory(libhello "lib/libhello")
add_subdirectory(C/string "lib/string")

# C++ libs
add_subdirectory(convert_utils "lib/convert_utils")

add_subdirectory(texture "lib/texture")

add_subdirectory(math "lib/math")

add_subdirectory(utils/opencv "lib/utils/opencv")

add_subdirectory(utils/opengl "lib/utils/opengl")

#add_subdirectory(utils/sfml "lib/utils/sfml")

add_subdirectory(time "lib/time")

add_subdirectory(cpu "lib/cpu")

add_subdirectory(vector "lib/vector")

add_subdirectory(thread "lib/thread")

if(EXTERNAL_LIB)
    add_dependencies(hello ${EXTERNAL_LIB})
    add_dependencies(convert_utils ${EXTERNAL_LIB})
    add_dependencies(texture ${EXTERNAL_LIB})
    add_dependencies(math ${EXTERNAL_LIB})
    add_dependencies(utils_opencv ${EXTERNAL_LIB})
    add_dependencies(utils_opengl ${EXTERNAL_LIB})
#   add_dependencies(utils_sfml ${EXTERNAL_LIB})
    add_dependencies(time ${EXTERNAL_LIB})
    add_dependencies(cpu ${EXTERNAL_LIB})
    add_dependencies(vector ${EXTERNAL_LIB})
    add_dependencies(thread ${EXTERNAL_LIB})
endif()

#   =================================================
#   =============    SOURCES FILES    ===============
#   =================================================

#   ================    FILES    ================
set(MAIN)

set(BENCH_ISPRIME src/bench/bench_isprime.cpp)

set(BENCH_VECTOR_COPY src/bench/bench_vector_copy.cpp)

set(BENCH_VECTOR_SORT src/bench/bench_vector_sort.cpp)

set(BENCH_DIGITS_COUNT src/bench/bench_digits_count.cpp)

set(BENCH_FIND_MAX src/bench/bench_find_max.cpp)

set(SRCS
    )

set(HEADERS
    )


#   ====================================================
#   ====================    LIBS    ====================
#   ====================================================

#set(MAIN_LLIBS Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Quick Qt${QT_VERSION_MAJOR}::Multimedia sfml-graphics sfml-window sfml-audio sfml-network sfml-system ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${OpenCV_LIBS})
set(MAIN_LLIBS sfml-graphics sfml-window sfml-audio sfml-network sfml-system ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${OpenCV_LIBS})

set(CUSTOM_LIBS texture convert_utils math utils_opencv utils_opengl time cpu vector thread)

#   ==================================================
#   ===============    EXEC FILES    =================
#   ==================================================

add_executable(bench_isprime ${BENCH_ISPRIME} ${SRCS} ${HEADERS})
target_link_libraries(bench_isprime PRIVATE ${MAIN_LLIBS} ${CUSTOM_LIBS} hello stdc++fs)
add_dependencies(bench_isprime ${CUSTOM_LIBS})

add_executable(bench_vector_copy ${BENCH_VECTOR_COPY} ${SRCS} ${HEADERS})
target_link_libraries(bench_vector_copy PRIVATE ${MAIN_LLIBS} ${CUSTOM_LIBS} hello stdc++fs)
add_dependencies(bench_vector_copy ${CUSTOM_LIBS})

add_executable(bench_vector_sort ${BENCH_VECTOR_SORT} ${SRCS} ${HEADERS})
target_link_libraries(bench_vector_sort PRIVATE ${MAIN_LLIBS} ${CUSTOM_LIBS} hello stdc++fs)
add_dependencies(bench_vector_sort ${CUSTOM_LIBS})

add_executable(bench_digits_count ${BENCH_DIGITS_COUNT} ${SRCS} ${HEADERS})
target_link_libraries(bench_digits_count PRIVATE ${MAIN_LLIBS} ${CUSTOM_LIBS} hello stdc++fs)
add_dependencies(bench_digits_count ${CUSTOM_LIBS})

add_executable(bench_find_max ${BENCH_FIND_MAX} ${SRCS} ${HEADERS})
target_link_libraries(bench_find_max PRIVATE ${MAIN_LLIBS} ${CUSTOM_LIBS} hello stdc++fs)
add_dependencies(bench_find_max ${CUSTOM_LIBS})


#   ======================================================
#   ====================    TESTS    =====================
#   ======================================================

enable_testing()

#Test getSchwarzschild
set(TEST_SCHWARZSCHILD test_getSchwarzschild)
add_executable(${TEST_SCHWARZSCHILD} src/test/math/test_getSchwarzschild.cpp)
if(EXTERNAL_LIB)
    add_dependencies(${TEST_SCHWARZSCHILD} ${EXTERNAL_LIB})
endif()
target_link_libraries(${TEST_SCHWARZSCHILD} ${Boost_LIBRARIES})
add_test(NAME ${TEST_SCHWARZSCHILD} COMMAND ${CMAKE_COMMAND} --build . --target ${TEST_SCHWARZSCHILD})

#Test getGravitationalAttraction
set(TEST_GA test_getGravitationalAttraction)
add_executable(${TEST_GA} src/test/math/test_getGravitationalAttraction.cpp)
if(EXTERNAL_LIB)
    add_dependencies(${TEST_GA} ${EXTERNAL_LIB})
endif()
target_link_libraries(${TEST_GA} ${Boost_LIBRARIES})
add_test(NAME ${TEST_GA} COMMAND ${TEST_GA})

#Test sphere
set(TEST_SPHERE test_sphere)
add_executable(${TEST_SPHERE} src/test/math/test_sphere.cpp)
if(EXTERNAL_LIB)
    add_dependencies(${TEST_SPHERE} ${EXTERNAL_LIB})
endif()
target_link_libraries(${TEST_SPHERE} ${Boost_LIBRARIES} quadmath)
add_test(NAME ${TEST_SPHERE} COMMAND ${TEST_SPHERE})

#Test cylinder
set(TEST_CYLINDER test_cylinder)
add_executable(${TEST_CYLINDER} src/test/math/test_cylinder.cpp)
if(EXTERNAL_LIB)
    add_dependencies(${TEST_CYLINDER} ${EXTERNAL_LIB})
endif()
target_link_libraries(${TEST_CYLINDER} ${Boost_LIBRARIES})
add_test(NAME ${TEST_CYLINDER} COMMAND ${TEST_CYLINDER})

#Test vector sort
set(TEST_VECTOR_SORT test_vector_sort)
add_executable(${TEST_VECTOR_SORT} src/test/vector/test_vector_sort.cpp)
if(EXTERNAL_LIB)
    add_dependencies(${TEST_VECTOR_SORT} ${EXTERNAL_LIB})
endif()
target_link_libraries(${TEST_VECTOR_SORT} ${Boost_LIBRARIES} vector)
add_test(NAME ${TEST_VECTOR_SORT} COMMAND ${TEST_VECTOR_SORT})

#Test vector copy
set(TEST_VECTOR_COPY test_vector_copy)
add_executable(${TEST_VECTOR_COPY} src/test/vector/test_vector_copy.cpp)
if(EXTERNAL_LIB)
    add_dependencies(${TEST_VECTOR_COPY} ${EXTERNAL_LIB})
endif()
target_link_libraries(${TEST_VECTOR_COPY} ${Boost_LIBRARIES} vector)
add_test(NAME ${TEST_VECTOR_COPY} COMMAND ${TEST_VECTOR_COPY})

#Test power
set(TEST_POWER test_power)
add_executable(${TEST_POWER} src/test/math/test_power.cpp)
if(EXTERNAL_LIB)
    add_dependencies(${TEST_POWER} ${EXTERNAL_LIB})
endif()
target_link_libraries(${TEST_POWER} ${Boost_LIBRARIES})
add_test(NAME ${TEST_POWER} COMMAND ${TEST_POWER})
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

#file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR})
#configure_file(font/Almond_Caramel.ttf ${CMAKE_BINARY_DIR}/font/Almond_Caramel.ttf COPYONLY)



#add_custom_command(
#    TARGET game
#    COMMENT "CP Texture"
#    POST_BUILD 
#    WORKING_DIRECTORY .
#    COMMAND cp -R -u -p "../texture/PIPOYA FREE RPG Character Sprites 32x32" "texture/PIPOYA FREE RPG Character Sprites 32x32"
#)

#   ================================================
#   ===============    UNSET VALUE    ===============
#   ================================================


#unset(CMAKE_BUILD_TYPE CACHE)
