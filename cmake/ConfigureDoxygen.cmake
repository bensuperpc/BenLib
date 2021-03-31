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

find_package(Doxygen)
#option(BUILD_DOCUMENTATION "Build doc" ON)
#option(BUILD_DOCUMENTATION "DOC" ${DOXYGEN_FOUND})

if (DOXYGEN_FOUND AND BUILD_DOCS_DOXYGEN)
    set(DOXYGEN_PROJECT_NAME "${CMAKE_PROJECT_NAME}")
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_XML NO)
    set(DOXYGEN_GENERATE_MAN NO)
    set(DOXYGEN_GENERATE_DOCBOOK NO)
    
    set(DOXYGEN_COLLABORATION_GRAPH YES)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_CLASS_DIAGRAMS YES)
    set(DOXYGEN_HIDE_UNDOC_RELATIONS NO)
    set(DOXYGEN_HAVE_DOT YES)
    set(DOXYGEN_CLASS_GRAPH YES)
    set(DOXYGEN_CALL_GRAPH YES)
    set(DOXYGEN_CALLER_GRAPH YES)
    set(DOXYGEN_COLLABORATION_GRAPH YES)
    set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
    set(DOXYGEN_EXTRACT_PRIVATE YES)
    set(DOXYGEN_EXTRACT_PACKAGE YES)
    set(DOXYGEN_EXTRACT_STATIC YES)
    set(DOXYGEN_SHOW_NAMESPACES YES)
    set(DOXYGEN_SHOW_FILES YES)
    set(DOXYGEN_SHOW_INCLUDE_FILES YES)
    set(DOXYGEN_SHOW_GROUPED_MEMB_INC YES)
    set(DOXYGEN_SHOW_USED_FILES YES)

    set(DOXYGEN_HTML_DYNAMIC_SECTIONS YES)
    set(GENERATE_TREEVIEW YES)
    set(DOXYGEN_CREATE_SUBDIRS YES)
    set(DOXYGEN_PDF_HYPERLINKS YES)
    set(DOXYGEN_GENERATE_AUTOGEN_DEF YES)
    set(DOXYGEN_SORT_GROUP_NAMES YES)
    set(DOXYGEN_DOT_NUM_THREADS 12)
    set(DOXYGEN_HIDE_IN_BODY_DOCS NO)
    set(DOXYGEN_STRIP_CODE_COMMENTS NO)
    set(DOXYGEN_BRIEF_MEMBER_DESC YES)
    set(DOXYGEN_TEMPLATE_RELATIONS YES)
    # Add CUDA and OpenCL files
    set(DOXYGEN_EXTENSION_MAPPING "cuh=C++;cu=C++;tpp=C++;cl=C")

    set(DOXYGEN_HTML_DYNAMIC_MENUS YES)
    # Disable some useless info
    set(DOXYGEN_QUIET YES)

    set(DOXYGEN_RECURSIVE YES)
    
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs")
    #set(DOXYGEN_INPUT "${CMAKE_CURRENT_BINARY_DIR}/README.md")
    #set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_BINARY_DIR}/README.md")

    set(DOXYGEN_FILE_PATTERNS *.c 
    *.cc 
    *.cxx 
    *.cpp 
    *.c++ 
    *.java 
    *.ii 
    *.ixx 
    *.ipp 
    *.i++ 
    *.inl 
    *.idl 
    *.ddl 
    *.odl 
    *.h 
    *.hh 
    *.hxx 
    *.hpp 
    *.tpp
    *.cu
    *.cuh
    *.cl   
    *.h++)

    set(DOXYGEN_INPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib;${PROJECT_SOURCE_DIR}/src")

    #USE_STAMP_FILE
    doxygen_add_docs(
        Doxygen ${DOXYGEN_INPUT_DIRECTORY} ALL COMMENT "Generate pages")
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/html DESTINATION docs)
else()
    message( "Doxygen need to be installed to generate the doxygen documentation" )
endif()
