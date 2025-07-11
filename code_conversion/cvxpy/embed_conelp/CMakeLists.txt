# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(embed_conelp C)

set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -ffast-math")
set(CMAKE_BUILD_TYPE Release)


# Set the C standard to C99
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Option to enable or disable unit test building
option(BUILD_TESTS "Build unit tests" ON)

# Try to find a BLAS library (OpenBLAS, ATLAS, MKL, etc.)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Collect all source files from the src/ directory
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS 
    src/base.c 
    src/blas.c
    src/sparse.c
    src/misc_solvers.c
    src/lapack.c
    src/dense.c
    src/coneprog.c
)

# Add the public header include path
include_directories(${PROJECT_SOURCE_DIR}/include)

# Create a static library target from source files
add_library(${PROJECT_NAME} SHARED ${LIB_SOURCES})

target_link_libraries(${PROJECT_NAME} m lapack blas)

# add_executable(test_main tests/test_main.c)
# target_link_libraries(test_main ${PROJECT_NAME})

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
    # include_directories(${CMAKE_SOURCE_DIR}/unity)

    # Collect all test source files
    file(GLOB TEST_FILES tests/c/*.c)

    # Create test executable by linking your library
    add_executable(test_main ${TEST_FILES})
    target_link_libraries(test_main ${PROJECT_NAME})

    # Register the test executable with CTest
    # add_test(NAME YourLibTests COMMAND unit_tests)
endif()
