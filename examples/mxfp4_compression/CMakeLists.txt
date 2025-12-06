cmake_minimum_required(VERSION 3.14)
project(MXFP4_Demo LANGUAGES C)

include(FetchContent)

FetchContent_Declare(
  bitsqueeze
  GIT_REPOSITORY https://github.com/DandinPower/BitSqueeze.git
  GIT_TAG        main
)

# Disable BitSqueeze tests
set(BITSQUEEZE_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(bitsqueeze)

add_executable(demo_app main.c)

target_link_libraries(demo_app PRIVATE BitSqueeze::bitsqueeze)