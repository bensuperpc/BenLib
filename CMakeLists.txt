cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    benlib
    VERSION 0.1.0
    DESCRIPTION "benlib"
    HOMEPAGE_URL "bensuperpc.org"
    LANGUAGES NONE
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Declare library ----

add_library(benlib_benlib INTERFACE)
add_library(benlib::benlib ALIAS benlib_benlib)

set_property(
    TARGET benlib_benlib PROPERTY
    EXPORT_NAME benlib
)

target_include_directories(
    benlib_benlib ${warning_guard}
    INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_compile_features(benlib_benlib INTERFACE cxx_std_20)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Examples ----

if(PROJECT_IS_TOP_LEVEL)
  option(BUILD_EXAMPLES "Build examples tree." "${benlib_DEVELOPER_MODE}")
  if(BUILD_EXAMPLES)
    add_subdirectory(example)
  endif()
endif()

# ---- Developer mode ----

if(NOT benlib_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of benlib"
  )
endif()

include(cmake/dev-mode.cmake)
