##############################################################
#   ____                                                     #
#  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___    #
#  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __|   #
#  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__    #
#  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___|   #
#                             |_|             |_|            #
##############################################################
#                                                            #
#  BenLib, 2021                                              #
#  Created: 01, April, 2021                                  #
#  Modified: 01, April, 2021                                 #
#  file: CMakeLists.txt                                      #
#  CMake                                                     #
#  Source:  
#  OS: ALL                                                   #
#  CPU: ALL                                                  #
#                                                            #
##############################################################

cmake_minimum_required(VERSION 3.12)
project(my_arduino LANGUAGES CXX C)


if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # Top-level project
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} -g3 -Og -ggdb3")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -march=native -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} -g3 -Og -ggdb3")
else()
    # Sub-level project
endif()


set(SRCS
    led/blink_number.cpp
    reboot/reboot_software.cpp
    usefull/arduino_type.cpp
    usefull/free_ram.cpp
    usefull/voltage.cpp
    lib/arduino_serial.cpp
    lib/arduino_time.cpp
    lib/arduino_io.cpp
    lib/arduino_math.cpp
    lib/arduino_is.cpp
    lib/arduino_string.cpp
	)
	
set(HEADERS
    led/blink_number.hpp
    reboot/reboot_software.hpp
    usefull/arduino_type.hpp
    usefull/free_ram.hpp
    usefull/voltage.hpp
    arduino_compatibility.hpp
    lib/arduino_serial.hpp
    lib/arduino_time.hpp
    lib/arduino_io.hpp
    lib/arduino_math.hpp
    lib/arduino_is.hpp
    lib/arduino_string.hpp
    )


#set_target_properties(my_cuda PROPERTIES PUBLIC_HEADER "${EDV_INCLUDE_FILES}")

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/lib")

add_library(my_arduino SHARED ${SRCS} ${HEADERS})
set_target_properties(my_arduino PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
set_target_properties(${PROJECT_NAME} PROPERTIES
    PUBLIC_HEADER "${HEADERS}"
)

#   ==================================================
#   ===============    EXECUTABLE    =================
#   ==================================================

# https://github.com/awakecoding/cmake/blob/master/Tests/Complex/Executable/CMakeLists.txt
#https://stackoverflow.com/a/40109487/10152334
#target_link_libraries(test1 PRIVATE )
#set_property(TARGET vec1d PROPERTY CUDA_ARCHITECTURES 52 60 61 62 70 72 75)
#set_source_files_properties(tests/basic.ino PROPERTIES COMPILE_FLAGS "-include \"arduino_compatibility.hpp\"")

# ======== Blink ========
set_source_files_properties(tests/Blink.ino PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY 1)
add_executable(Blink tests/main_arduino.cpp tests/Blink.ino) #arduino_compatibility.hpp
set_target_properties(Blink PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
target_compile_options(Blink PRIVATE -include "arduino_compatibility.hpp")
target_compile_definitions(Blink PRIVATE FILE_ARG=tests/Blink.ino LED_BUILTIN=1)
target_link_libraries(Blink my_arduino) # Optional

# ======== BlinkWithoutDelay ========
set_source_files_properties(tests/BlinkWithoutDelay.ino PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY 1)
add_executable(BlinkWithoutDelay tests/main_arduino.cpp tests/BlinkWithoutDelay.ino)
set_target_properties(BlinkWithoutDelay PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
target_compile_options(BlinkWithoutDelay PRIVATE -include "arduino_compatibility.hpp")
target_compile_definitions(BlinkWithoutDelay PRIVATE FILE_ARG=tests/BlinkWithoutDelay.ino LED_BUILTIN=1)
target_link_libraries(BlinkWithoutDelay my_arduino)

# ======== AnalogReadSerial ========
set_source_files_properties(tests/AnalogReadSerial.ino PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY 1)
add_executable(AnalogReadSerial tests/main_arduino.cpp tests/AnalogReadSerial.ino)
set_target_properties(AnalogReadSerial PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
target_compile_options(AnalogReadSerial PRIVATE -include "arduino_compatibility.hpp")
target_compile_definitions(AnalogReadSerial PRIVATE FILE_ARG=tests/AnalogReadSerial.ino)
target_link_libraries(AnalogReadSerial my_arduino)

# ======== ASCIITable ========
set_source_files_properties(tests/Communication/ASCIITable.ino PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY 1)
add_executable(ASCIITable tests/main_arduino.cpp tests/Communication/ASCIITable.ino)
set_target_properties(ASCIITable PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
target_compile_options(ASCIITable PRIVATE -include "arduino_compatibility.hpp")
target_compile_definitions(ASCIITable PRIVATE FILE_ARG=tests/Communication/ASCIITable.ino)
target_link_libraries(ASCIITable my_arduino)

# ======== Arrays ========
set_source_files_properties(tests/Control_Structures/Arrays.ino PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY 1)
add_executable(Arrays tests/main_arduino.cpp tests/Control_Structures/Arrays.ino)
set_target_properties(Arrays PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
target_compile_options(Arrays PRIVATE -include "arduino_compatibility.hpp")
target_compile_definitions(Arrays PRIVATE FILE_ARG=tests/Control_Structures/Arrays.ino)
target_link_libraries(Arrays my_arduino)

# ======== For_Loop_Iteration ========
set_source_files_properties(tests/Control_Structures/For_Loop_Iteration.ino PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY 1)
add_executable(For_Loop_Iteration tests/main_arduino.cpp tests/Control_Structures/For_Loop_Iteration.ino)
set_target_properties(For_Loop_Iteration PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
target_compile_options(For_Loop_Iteration PRIVATE -include "arduino_compatibility.hpp")
target_compile_definitions(For_Loop_Iteration PRIVATE FILE_ARG=tests/Control_Structures/For_Loop_Iteration.ino)
target_link_libraries(For_Loop_Iteration my_arduino)

# ======== SwitchCase ========
set_source_files_properties(tests/Control_Structures/SwitchCase.ino PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY 1)
add_executable(SwitchCase tests/main_arduino.cpp tests/Control_Structures/SwitchCase.ino)
set_target_properties(SwitchCase PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
target_compile_options(SwitchCase PRIVATE -include "arduino_compatibility.hpp")
target_compile_definitions(SwitchCase PRIVATE FILE_ARG=tests/Control_Structures/SwitchCase.ino)
target_link_libraries(SwitchCase my_arduino)

# ======== SwitchCase2 ========
set_source_files_properties(tests/Control_Structures/SwitchCase2.ino PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY 1)
add_executable(SwitchCase2 tests/main_arduino.cpp tests/Control_Structures/SwitchCase2.ino)
set_target_properties(SwitchCase2 PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
target_compile_options(SwitchCase2 PRIVATE -include "arduino_compatibility.hpp")
target_compile_definitions(SwitchCase2 PRIVATE FILE_ARG=tests/Control_Structures/SwitchCase2.ino)
target_link_libraries(SwitchCase2 my_arduino)

# ======== WhileStatementConditional ========
set_source_files_properties(tests/Control_Structures/WhileStatementConditional.ino PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY 1)
add_executable(WhileStatementConditional tests/main_arduino.cpp tests/Control_Structures/WhileStatementConditional.ino)
set_target_properties(WhileStatementConditional PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
target_compile_options(WhileStatementConditional PRIVATE -include "arduino_compatibility.hpp")
target_compile_definitions(WhileStatementConditional PRIVATE FILE_ARG=tests/Control_Structures/WhileStatementConditional.ino)
target_link_libraries(WhileStatementConditional my_arduino)


# ======== CharacterAnalysis ========
set_source_files_properties(tests/Strings/CharacterAnalysis.ino PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY 1)
add_executable(CharacterAnalysis tests/main_arduino.cpp tests/Strings/CharacterAnalysis.ino)
set_target_properties(CharacterAnalysis PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
target_compile_options(CharacterAnalysis PRIVATE -include "arduino_compatibility.hpp")
target_compile_definitions(CharacterAnalysis PRIVATE FILE_ARG=tests/Strings/CharacterAnalysis.ino)
target_link_libraries(CharacterAnalysis my_arduino)

# ======== StringAdditionOperator ========
set_source_files_properties(tests/Strings/StringAdditionOperator.ino PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY 1)
add_executable(StringAdditionOperator tests/main_arduino.cpp tests/Strings/StringAdditionOperator.ino)
set_target_properties(StringAdditionOperator PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
target_compile_options(StringAdditionOperator PRIVATE -include "arduino_compatibility.hpp")
target_compile_definitions(StringAdditionOperator PRIVATE FILE_ARG=tests/Strings/StringAdditionOperator.ino)
target_link_libraries(StringAdditionOperator my_arduino)

# ======== StringAppendOperator ========
set_source_files_properties(tests/Strings/StringAppendOperator.ino PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY 1)
add_executable(StringAppendOperator tests/main_arduino.cpp tests/Strings/StringAppendOperator.ino)
set_target_properties(StringAppendOperator PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
target_compile_options(StringAppendOperator PRIVATE -include "arduino_compatibility.hpp")
target_compile_definitions(StringAppendOperator PRIVATE FILE_ARG=tests/Strings/StringAppendOperator.ino)
target_link_libraries(StringAppendOperator my_arduino)

# ======== StringCaseChanges ========
set_source_files_properties(tests/Strings/StringCaseChanges.ino PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY 1)
add_executable(StringCaseChanges tests/main_arduino.cpp tests/Strings/StringCaseChanges.ino)
set_target_properties(StringCaseChanges PROPERTIES AUTOUIC OFF AUTOMOC OFF AUTORCC OFF)
target_compile_options(StringCaseChanges PRIVATE -include "arduino_compatibility.hpp")
target_compile_definitions(StringCaseChanges PRIVATE FILE_ARG=tests/Strings/StringCaseChanges.ino)
target_link_libraries(StringCaseChanges my_arduino)


#   ================================================
#   ================    INSTALL    =================
#   ================================================

install(TARGETS ${PROJECT_NAME} 
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # Nedd more 1 target
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Nedd more 1 target
#    PRIVATE_HEADER DESTINATION source
    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}"
)
