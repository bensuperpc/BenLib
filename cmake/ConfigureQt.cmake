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


if(NOT DEFINED CMAKE_AUTOUIC)
    set(CMAKE_AUTOUIC ON)
endif()
if(NOT DEFINED CMAKE_AUTOMOC)
    set(CMAKE_AUTOMOC ON)
endif()
if(NOT DEFINED CMAKE_AUTORCC)
    set(CMAKE_AUTORCC ON)
endif()

find_package(QT NAMES Qt6 Qt5 COMPONENTS Core Quick Multimedia)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core Quick Multimedia)
find_package(Qt${QT_VERSION_MAJOR}QuickCompiler)
#qtquick_compiler_add_resources(RESOURCES example.qrc)
#qt5_use_modules(myapp Quick Widgets Core Gui Multimedia Network)

#find_package(Qt5 COMPONENTS Quick Widgets Core Gui Multimedia Network REQUIRED)

if (QT_FOUND)
    message(STATUS "QT: FOUND")
    include_directories(${QT_INCLUDES})
else()
    message(STATUS "QT: NOT FOUND")
endif()
