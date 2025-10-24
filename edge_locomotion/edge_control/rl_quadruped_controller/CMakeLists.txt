cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
project(rl_quadruped_controller)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Some LibTorch distributions reference libkineto, which may not be present
# on certain platforms (e.g., aarch64 prebuilt CUDA bundles). Filter it out
# from TORCH_LIBRARIES if it appears to avoid Make errors for a missing file.
if (TORCH_LIBRARIES)
        set(_filtered_torch_libs ${TORCH_LIBRARIES})
        list(FILTER _filtered_torch_libs EXCLUDE REGEX ".*/libkineto\\.(a|so)(\\.\\d+)?$")
        if (NOT _filtered_torch_libs STREQUAL "${TORCH_LIBRARIES}")
                message(STATUS "Filtered out libkineto from TORCH_LIBRARIES")
                set(TORCH_LIBRARIES ${_filtered_torch_libs})
        endif ()
endif ()

# Determine LibTorch library directory for RPATH
set(LIBTORCH_LIB_DIR "")
foreach(_lib IN LISTS TORCH_LIBRARIES)
        if (_lib MATCHES ".*/libtorch\\.(so|dylib|a)(\\..*)?$")
                get_filename_component(_dir "${_lib}" DIRECTORY)
                set(LIBTORCH_LIB_DIR "${_dir}")
        endif ()
endforeach ()
if (NOT LIBTORCH_LIB_DIR)
        # Fallback: derive from Torch_DIR -> <prefix>/lib
        get_filename_component(_torch_share "${Torch_DIR}" DIRECTORY)    # .../share/cmake
        get_filename_component(_torch_share_root "${_torch_share}" DIRECTORY) # .../share
        get_filename_component(_torch_prefix "${_torch_share_root}" DIRECTORY) # prefix
        set(LIBTORCH_LIB_DIR "${_torch_prefix}/lib")
endif ()

# Prepare RPATHs so the plugin can locate LibTorch and CUDA at runtime
set(_rlc_rpaths)
list(APPEND _rlc_rpaths "${LIBTORCH_LIB_DIR}")
if (EXISTS "/usr/local/cuda/lib64")
        list(APPEND _rlc_rpaths "/usr/local/cuda/lib64")
endif ()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)

find_package(yaml-cpp REQUIRED)

set(dependencies
        pluginlib
        rcpputils
        controller_interface
        realtime_tools
        control_input_msgs
        kdl_parser
        ament_index_cpp
        controller_common
)

# find dependencies
foreach (Dependency IN ITEMS ${dependencies})
    find_package(${Dependency} REQUIRED)
endforeach ()

add_library(${PROJECT_NAME} SHARED
        src/RlQuadrupedController.cpp

        src/common/ObservationBuffer.cpp

        src/FSM/StateFixedStand.cpp
        src/FSM/StateRL.cpp

        src/control/LowPassFilter.cpp
        src/control/Estimator.cpp

        src/robot/QuadrupedRobot.cpp
        src/robot/RobotLeg.cpp

)
set_target_properties(${PROJECT_NAME}
        PROPERTIES
        BUILD_RPATH "${_rlc_rpaths}"
        INSTALL_RPATH "${_rlc_rpaths}"
)
target_include_directories(${PROJECT_NAME}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
        PRIVATE
        ${YAML_CPP_INCLUDE_DIR}
        src)
target_link_libraries(${PROJECT_NAME} PUBLIC
        "${TORCH_LIBRARIES}"
        yaml-cpp
)
ament_target_dependencies(
        ${PROJECT_NAME} PUBLIC
        ${dependencies}
)

pluginlib_export_plugin_description_file(controller_interface rl_quadruped_controller.xml)

install(
        DIRECTORY include/
        DESTINATION include/${PROJECT_NAME}
)

install(
        TARGETS ${PROJECT_NAME}
        EXPORT export_${PROJECT_NAME}
        ARCHIVE DESTINATION lib/${PROJECT_NAME}
        LIBRARY DESTINATION lib/${PROJECT_NAME}
        RUNTIME DESTINATION bin
)

install(
        DIRECTORY launch
        DESTINATION share/${PROJECT_NAME}/
)

ament_export_dependencies(${dependencies})
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # comment the line when a copyright and license is added to all source files
    set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # comment the line when this package is in a git repo and when
    # a copyright and license is added to all source files
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

ament_package()
