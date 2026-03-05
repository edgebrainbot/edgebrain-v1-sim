cmake_minimum_required(VERSION 3.8)
project(edge_trigger)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- dependencies ----
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(edge_config REQUIRED)
find_package(edge_msgs REQUIRED)
find_package(edge_ros_interfaces REQUIRED)

# ---- executable node ----
add_executable(trigger
  src/trigger.cpp
)

target_include_directories(trigger PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(trigger
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  edge_config
  edge_msgs
  edge_ros_interfaces
)

# ---------------------------------------------------------
# Test node: test_trigger_node (manual runtime test)
# ---------------------------------------------------------
add_executable(test_trigger
  test/trigger_test.cpp
)

target_include_directories(test_trigger PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(test_trigger
  rclcpp
  std_msgs
  edge_msgs
  edge_ros_interfaces
)

# ---------------------------------------------------------
# Installation
# ---------------------------------------------------------
install(TARGETS
  trigger
  test_trigger
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# ---- optional test lint ----
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()


ament_package()
