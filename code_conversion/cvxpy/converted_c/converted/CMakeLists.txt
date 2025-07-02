# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(embeded_cvxopt C)

# Set the C standard to C99
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Option to enable or disable unit test building
option(BUILD_TESTS "Build unit tests" OFF)

# Try to find a BLAS library (OpenBLAS, ATLAS, MKL, etc.)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Collect all source files from the src/ directory
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS 
    tests/test_base.c 
    tests/test_blas.c
    tests/test_sparse.c
    tests/test_misc_solvers.c
    tests/test_lapack.c
    tests/test_dense.c
)

# Add the public header include path
include_directories(${PROJECT_SOURCE_DIR}/include)

# Create a static library target from source files
add_library(base SHARED ${LIB_SOURCES})

target_link_libraries(base m lapack blas)

add_executable(test_main tests/test_main.c)
target_link_libraries(test_main base)

message(STATUS "------ Debug Info ------")
message(STATUS "CMAKE_SOURCE_DIR        = ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR        = ${CMAKE_BINARY_DIR}")
message(STATUS "------------------------")


# Conditional block to build tests if enabled
if(BUILD_TESTS)
    # Enable CTest support for test discovery
    enable_testing()

    # Add Unity framework directory to the include path
    include_directories(${CMAKE_SOURCE_DIR}/unity)

    # Gather Unity source file(s)
    file(GLOB_RECURSE UNITY_SRC unity/*.c)

    # Collect all test source files
    file(GLOB_RECURSE TEST_SOURCES tests/*.c)

    # Create test executable by linking your library and Unity
    add_executable(unit_tests ${TEST_SOURCES} ${UNITY_SRC})
    target_link_libraries(unit_tests yourlib)

    # Register the test executable with CTest
    add_test(NAME YourLibTests COMMAND unit_tests)
endif()
