# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(ecvxcone C)

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

# set(OPENBLAS_PATH "${CMAKE_SOURCE_DIR}/../../../../../../anaconda3/lib/python3.12/site-packages/cvxopt/../cvxopt.libs/libopenblas-r0-e9c9f581.3.15.so")
# set(LAPACK_PATH "${CMAKE_SOURCE_DIR}/../../../../../../anaconda3/lib/python3.12/site-packages/cvxopt/../cvxopt.libs/libopenblas-r0-e9c9f581.3.15.so")
# if(NOT EXISTS ${OPENBLAS_PATH})
    # message(FATAL_ERROR "OpenBLAS not found at ${OPENBLAS_PATH}")
# endif()

# 你可以根据需要调整路径方式，或通过环境变量传入
# message(STATUS "Using OpenBLAS from: ${OPENBLAS_PATH}")
# set(BLAS_LIBRARIES ${OPENBLAS_PATH})
# set(LAPACK_LIBRARIES ${OPENBLAS_PATH})

# Collect all source files from the src/ directory
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS 
    src/base.c 
    src/blas.c
    src/debug.c
    src/sparse.c
    src/misc.c
    src/lapack.c
    src/dense.c
    src/solver.c
    src/utils.c
)

# Add the public header include path
include_directories(${PROJECT_SOURCE_DIR}/include)

# Create a static library target from source files
add_library(${PROJECT_NAME} SHARED ${LIB_SOURCES})

# target_compile_definitions(${PROJECT_NAME} PRIVATE BLAS_NO_UNDERSCORE)

target_link_libraries(${PROJECT_NAME} m lapack blas)
# target_link_libraries(${PROJECT_NAME} m ${LAPACK_LIBRARIES})

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

    # 添加 RPATH，告诉它运行时去哪里找动态库
    # set_target_properties(test_main PROPERTIES
    #     BUILD_RPATH "$ENV{HOME}/anaconda3/lib/python3.12/site-packages/cvxopt/../cvxopt.libs"
    #     INSTALL_RPATH "$ENV{HOME}/anaconda3/lib/python3.12/site-packages/cvxopt/../cvxopt.libs"
    # )

    # target_compile_definitions(test_main PRIVATE BLAS_NO_UNDERSCORE)
    # Register the test executable with CTest
    # add_test(NAME YourLibTests COMMAND unit_tests)
endif()
