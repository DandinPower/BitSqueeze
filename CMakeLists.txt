cmake_minimum_required(VERSION 3.10)

project(BitSqueeze C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------------------
# 1. Compiler Flags and Options
# -------------------------------------------------------------
# For POSIX (clock_gettime)
add_definitions(-D_POSIX_C_SOURCE=199309L)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall 
        -Wextra 
        -Wpedantic 
        -Wno-misleading-indentation 
        -Wno-format
        -O3
    )
endif()

# -------------------------------------------------------------
# 2. Dependencies (OpenMP and Math)
# -------------------------------------------------------------
find_package(OpenMP)

# -------------------------------------------------------------
# 3. Build the Core Library
# -------------------------------------------------------------
file(GLOB_RECURSE LIB_SOURCES "src/*.c")

add_library(bitsqueeze STATIC ${LIB_SOURCES})

target_include_directories(bitsqueeze PUBLIC include)

target_link_libraries(bitsqueeze PUBLIC m)

if(OpenMP_C_FOUND)
    target_link_libraries(bitsqueeze PUBLIC OpenMP::OpenMP_C)
endif()

# -------------------------------------------------------------
# 4. Build Tests
# -------------------------------------------------------------
enable_testing()

file(GLOB TEST_SOURCES "test/*.c")

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE bitsqueeze)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()