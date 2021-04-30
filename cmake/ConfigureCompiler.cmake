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
#  Created: 29, March, 2021                                  #
#  Modified: 29, March, 2021                                 #
#  file: ConfigureBoost.cmake                                #
#  CMake                                                     #
#  Source:                                                   #
#  OS: ALL                                                   #
#  CPU: ALL                                                  #
#                                                            #
##############################################################


# Enable CXX standard required
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Enable C standard required
#set(CMAKE_C_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_C_EXTENSIONS OFF)

message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}.")

set(AUTODETECT_CXX_MAX_VERSION "Detect automatically CXX standard Max supported version by compiler" ON)
set(AUTODETECT_C_MAX_VERSION "Detect automatically C standard Max supported version by compiler" ON)
# Shared libs, decrease bin size
set(BUILD_SHARED_LIBS "Build shared (dynamic) libraries" ON)
#option(CLANG_ENABLE_CODE_COVERAGE ON)
set(CLANG_THREADSANIZER OFF)


#Set the highest C++ standard supported by the compiler
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_C_STANDARD 11)

#=== C VERSION CHECK ===
# Set the highest C standard supported by the compiler

if(NOT CMAKE_C_STANDARD AND AUTODETECT_C_MAX_VERSION)
    include(CheckCCompilerFlag)
    CHECK_C_COMPILER_FLAG("-std=c23" COMPILER_SUPPORTS_C23)
    CHECK_C_COMPILER_FLAG("-std=c20" COMPILER_SUPPORTS_C20)
    CHECK_C_COMPILER_FLAG("-std=c18" COMPILER_SUPPORTS_C18)
    CHECK_C_COMPILER_FLAG("-std=c17" COMPILER_SUPPORTS_C17) # Not working with CMake 3.19
    CHECK_C_COMPILER_FLAG("-std=c11" COMPILER_SUPPORTS_C11)
    CHECK_C_COMPILER_FLAG("-std=c99" COMPILER_SUPPORTS_C99)
    CHECK_C_COMPILER_FLAG("-std=c90" COMPILER_SUPPORTS_C90)
    CHECK_C_COMPILER_FLAG("-std=c89" COMPILER_SUPPORTS_C89)
    
    if(COMPILER_SUPPORTS_C11)
        message(STATUS "C11: OK")
        set(CMAKE_C_STANDARD 11)
    elseif(COMPILER_SUPPORTS_C99)
        message(STATUS "C99: OK")
        set(CMAKE_C_STANDARD 99)
    elseif(COMPILER_SUPPORTS_C90)
        message(STATUS "C90: OK")
        set(CMAKE_C_STANDARD 90)
    elseif(COMPILER_SUPPORTS_C89)
        message(STATUS "C89: OK")
        set(CMAKE_C_STANDARD 89)
    else()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 or above support. Please use a different C++ compiler.")
    endif()
endif()


#=== C++ VERSION CHECK ===
if(NOT CMAKE_CXX_STANDARD AND AUTODETECT_CXX_MAX_VERSION)
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-std=c++23" COMPILER_SUPPORTS_CXX23)
    CHECK_CXX_COMPILER_FLAG("-std=c++20" COMPILER_SUPPORTS_CXX20)
    CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
    CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
    CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
    CHECK_CXX_COMPILER_FLAG("-std=c++03" COMPILER_SUPPORTS_CXX03)
    CHECK_CXX_COMPILER_FLAG("-std=c++98" COMPILER_SUPPORTS_CXX98)

    if(COMPILER_SUPPORTS_CXX20)
        set(CMAKE_CXX_STANDARD 20)
        message(STATUS "C++20: OK")
    elseif(COMPILER_SUPPORTS_CXX17)
        set(CMAKE_CXX_STANDARD 17)
        message(STATUS "C++17: OK")
    elseif(COMPILER_SUPPORTS_CXX14)
        set(CMAKE_CXX_STANDARD 14)
        message(WARNING "C++14: Error")
        message(FATAL_ERROR "C++14 is old, please use newer compiler.")
    elseif(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_STANDARD 11)
        message(WARNING "C++11: OK")
        message(FATAL_ERROR "C++11 is old, please use newer compiler.")
    elseif(COMPILER_SUPPORTS_CXX03)
        set(CMAKE_CXX_STANDARD 03)
        message(WARNING "C++03: OK")
        message(FATAL_ERROR "C++03 is old, please use newer compiler.")
    elseif(COMPILER_SUPPORTS_CXX98)
        set(CMAKE_CXX_STANDARD 98)
        message(WARNING "C++98: OK")
        message(FATAL_ERROR "C++98 is old, please use newer compiler.")
    else()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 or above support. Please use a different C++ compiler.")
    endif()

    # Compiler-specific C++11 activation.
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        # require at least gcc 4.8
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
            message(FATAL_ERROR "GCC version must be at least 4.8!")
            set(CMAKE_CXX_STANDARD 14)
        endif()
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        # require at least clang 3.2
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.2)
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

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0)
    set(SEC_COMPILER_REL " ${SEC_COMPILER_REL} -fstack-clash-protection")
endif()


#=== SECURITY RELEASE FLAGS ===
set(SEC_COMPILER_REL " ${SEC_COMPILER_REL} -fstack-protector-all -Werror=format-security -fstack-protector-strong -fexceptions -D_FORTIFY_SOURCE=2 -fPIE")

#=== C FLAGS ===
set(WARNINGS_COMPILER_C "-Wall -Wpedantic -Wextra -Wstrict-prototypes -Wmissing-prototypes -Wfloat-equal -Wundef -Wshadow -Wpointer-arith -Wstrict-overflow=5 -Wswitch-default -Wunreachable-code -Wcast-align")

set(CMAKE_C_FLAGS                "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections ${WARNINGS_COMPILER_C} -pipe ")
set(CMAKE_C_FLAGS_RELEASE        "${CMAKE_C_FLAGS_RELEASE} -O3 ${SEC_COMPILER_REL}")
set(CMAKE_C_FLAGS_MINSIZEREL     "${CMAKE_C_FLAGS_MINSIZEREL} -Os ${SEC_COMPILER_REL}")
set(CMAKE_C_FLAGS_DEBUG          "${CMAKE_C_FLAGS_DEBUG} -g3 -Og -ggdb3 -v") # Remove -v
set(CMAKE_C_FLAGS_COVERAGE "${CMAKE_C_FLAGS_DEBUG}")


if(NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_C_FLAGS                "${CMAKE_C_FLAGS} -march=native")
endif()

#=== CXX FLAGS ===
#-Wold-style-cast -Wdouble-promotion -fstack-usage -Wpadded

#add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wshadow>")


set(WARNINGS_COMPILER_CXX "${WARNINGS_COMPILER_CXX} -Wall -Wextra -Wpedantic -Wshadow -Wmissing-declarations -Wundef -Wstack-protector -Wno-unused-parameter")
set(WARNINGS_COMPILER_CXX "${WARNINGS_COMPILER_CXX} -Wmissing-include-dirs -Wmissing-noreturn -Wimport -Winit-self -Winvalid-pch -Wstrict-aliasing=2 -Wswitch-default -Wunreachable-code -Wunused")
set(WARNINGS_COMPILER_CXX "${WARNINGS_COMPILER_CXX} -Woverloaded-virtual  -Wdisabled-optimization -Winline -Wredundant-decls -Wsign-conversion -Wformat-nonliteral -Wformat-security")
set(WARNINGS_COMPILER_CXX "${WARNINGS_COMPILER_CXX} -Wwrite-strings -Wcast-align -Wcast-qual -Wfloat-equal -Wvariadic-macros -Wpacked -Wpointer-arith -Weffc++ -Wformat=2 -Wfloat-equal")
set(WARNINGS_COMPILER_CXX "${WARNINGS_COMPILER_CXX} -Wnull-dereference")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(WARNINGS_COMPILER_CXX "${WARNINGS_COMPILER_CXX} -Wabi=11 -Wduplicated-branches -Wduplicated-cond -Wlogical-op")
endif()


# Removed : -Wconversion -Wuseless-cast -rdynamic
#-ftime-report -static -lrt -pthread -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -flto

set(CMAKE_CXX_FLAGS                " ${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections ${WARNINGS_COMPILER_CXX} -pipe")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS                " ${CMAKE_CXX_FLAGS} -lstdc++fs")
endif()

set(CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_MINSIZEREL} -Os ${SEC_COMPILER_REL} -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3 ${SEC_COMPILER_REL} -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} -g3 -Og -ggdb3 -v") # Remove -v -pg
set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_C_FLAGS_DEBUG}")

if(NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_CXX_FLAGS                "${CMAKE_CXX_FLAGS} -march=native")
endif()


#gcov
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s -fPIC -Wl,-z,now -Wl,-z,relro -Wl,--sort-common,--as-needed,--gc-sections,--strip-all,--allow-multiple-definition -Wl,-rpath,../lib -Wl,-rpath,../external/lib -Wl,-rpath,../../lib ")

if (NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
endif()

# Include to config Ninja
include(ConfigureNinja)

#add_compile_options(
#    "$<$<CXX_COMPILER_ID:Clang>:>"
#    "$<$<CXX_COMPILER_ID:Gnu>:>"
#    "$<$<CXX_COMPILER_ID:MSVC>:>"
#    )
#add_compile_options(
#    $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CXX_FLAGS}>
#    $<$<COMPILE_LANGUAGE:C>:${c_flags}>
#    )
#target_compile_options(my-target
#  PRIVATE
#    $<$<CXX_COMPILER_ID:Gnu>:
#      # g++ warning flags
#    >
#    $<$<CXX_COMPILER_ID:Clang>:
#      # clang warning flags
#    >
#    $<$<CXX_COMPILER_ID:MSVC>:
#      # MSVC warning flags
#    >
#)

#CLANG_THREADSANIZER
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    #-fsanitize=address -fsanitize-memory-track-origins
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    #if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.0)
        set(CMAKE_CXX_FLAGS                 " ${CMAKE_CXX_FLAGS} -fuse-ld=gold")
        set(CMAKE_C_FLAGS                   " ${CMAKE_C_FLAGS} -fuse-ld=gold")
        set(CMAKE_EXE_LINKER_FLAGS          " ${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
    else()
        set(CMAKE_CXX_FLAGS                 " ${CMAKE_CXX_FLAGS} -flto=thin")
        set(CMAKE_C_FLAGS                   " ${CMAKE_C_FLAGS} -flto=thin")
        set(CMAKE_EXE_LINKER_FLAGS                " ${CMAKE_EXE_LINKER_FLAGS} -flto=thin -Wl,--thinlto-jobs=all")
    endif()
    #endif()

    if (CODE_COVERAGE OR CMAKE_BUILD_TYPE STREQUAL "Coverage")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage -fprofile-instr-generate -fcoverage-mapping")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.2.0)
        set(CMAKE_EXE_LINKER_FLAGS          " ${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
        set(CMAKE_CXX_FLAGS                 " ${CMAKE_CXX_FLAGS} -fuse-ld=gold")
        set(CMAKE_C_FLAGS                   " ${CMAKE_C_FLAGS} -fuse-ld=gold")
    else()
        set(CMAKE_CXX_FLAGS                " ${CMAKE_CXX_FLAGS} -flto")
        set(CMAKE_C_FLAGS                " ${CMAKE_C_FLAGS} -flto")
        set(CMAKE_EXE_LINKER_FLAGS                " ${CMAKE_EXE_LINKER_FLAGS} -flto")
    endif()

    if (CODE_COVERAGE OR CMAKE_BUILD_TYPE STREQUAL "Coverage")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage")
    endif()
else()
endif()

# llvm-cov