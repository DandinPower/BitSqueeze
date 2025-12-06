cmake_minimum_required(VERSION 3.14)

project(BitSqueeze VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BITSQUEEZE_BUILD_TESTS "Build BitSqueeze tests" ON)

# -------------------------------------------------------------
# 3. Dependencies
# -------------------------------------------------------------
find_package(OpenMP)

# -------------------------------------------------------------
# 4. Build the Core Library
# -------------------------------------------------------------
file(GLOB_RECURSE LIB_SOURCES "src/*.c")

add_library(bitsqueeze STATIC ${LIB_SOURCES})

add_library(BitSqueeze::bitsqueeze ALIAS bitsqueeze)

target_compile_definitions(bitsqueeze PRIVATE _POSIX_C_SOURCE=199309L)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bitsqueeze PRIVATE
        -Wall 
        -Wextra 
        -Wpedantic 
        -Wno-misleading-indentation 
        -Wno-format
        -O3
    )
endif()

# 7. Define Includes using Generator Expressions
target_include_directories(bitsqueeze PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(bitsqueeze PUBLIC m)

if(OpenMP_C_FOUND)
    target_link_libraries(bitsqueeze PUBLIC OpenMP::OpenMP_C)
endif()

# -------------------------------------------------------------
# 8. Installation Rules (Enable `find_package` support)
# -------------------------------------------------------------
install(TARGETS bitsqueeze
    EXPORT BitSqueezeTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT BitSqueezeTargets
    FILE BitSqueezeTargets.cmake
    NAMESPACE BitSqueeze::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BitSqueeze
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "BitSqueezeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BitSqueezeConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/BitSqueezeConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BitSqueeze
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/BitSqueezeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/BitSqueezeConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BitSqueeze
)

# -------------------------------------------------------------
# 9. Build Tests (Only if this is the top project or requested)
# -------------------------------------------------------------
if(BITSQUEEZE_BUILD_TESTS)
    enable_testing()

    file(GLOB TEST_SOURCES "test/*.c")

    foreach(test_src ${TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} PRIVATE BitSqueeze::bitsqueeze)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()