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

#OpenSSL::SSL OpenSSL::Crypto

find_package(OpenSSL QUIET)

if (OPENSSL_FOUND)
    include_directories( ${OPENSSL_INCLUDE_DIR})
    #target_include_directories(${PROJECT_NAME} PUBLIC ${OPENSSL_INCLUDE_DIR})
    link_directories(${OPENSSL_LIBRARIES})
    message(STATUS "OPENSSL: FOUND")
    message(STATUS "Found OpenSSL: ${OPENSSL_VERSION}")
else()
    message(STATUS "OPENSSL: NOT FOUND")
endif()