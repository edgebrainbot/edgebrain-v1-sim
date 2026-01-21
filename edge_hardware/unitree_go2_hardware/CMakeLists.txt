cmake_minimum_required(VERSION 3.8)
project(unitree_go2_hardware)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# Generate symbols for IDE indexer
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PKG_DEPENDENCIES
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(backward_ros REQUIRED)
find_package(unitree_sdk2 REQUIRED)

foreach (Dependency IN ITEMS ${PKG_DEPENDENCIES})
  find_package(${Dependency} REQUIRED)
endforeach ()
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

include(GNUInstallDirs)
add_library(
  ${PROJECT_NAME}
  SHARED
  src/go2_hardware_interface.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
  PRIVATE src
)
target_link_libraries(${PROJECT_NAME} unitree_sdk2)
ament_target_dependencies(
        ${PROJECT_NAME}
        ${PKG_DEPENDENCIES}
)

pluginlib_export_plugin_description_file(hardware_interface go2_hardware_interface.xml)


add_executable(test_crc32 test/test_crc32.cpp)
target_include_directories(test_crc32 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
) 
target_link_libraries(test_crc32 unitree_sdk2)

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

# ---------------- Installation ----------------
# Install library
install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install test executable
install(TARGETS test_crc32
  DESTINATION lib/${PROJECT_NAME}
)

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

ament_export_dependencies(${dependencies})
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
