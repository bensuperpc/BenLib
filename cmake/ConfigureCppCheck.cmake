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


# https://stackoverflow.com/a/63840110/10152334

#set(ENABLE_CODE_ANALYSIS "Run code analysis" OFF)

message(STATUS "ENABLE_CODE_ANALYSIS: ${ENABLE_CODE_ANALYSIS}")

if(ENABLE_CODE_ANALYSIS)
    find_program(cppcheck cppcheck)
    message(STATUS "cppcheck                   ${cppcheck}")
    if(NOT (cppcheck MATCHES "NOTFOUND"))
        set(CMAKE_CXX_CPPCHECK "${cppcheck}"
            "--enable=all"
            "--inconclusive"
            "--inline-suppr"
            "--quiet"
            "--suppress=unmatchedSuppression"
            "--suppress=unusedFunction"
            "--template='{file}:{line}: warning: {id} ({severity}): {message}'")
    endif()
endif(ENABLE_CODE_ANALYSIS)