cmake_minimum_required(VERSION 3.8)
project(edge_dummy_env)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(edge_msgs REQUIRED)
find_package(edge_config REQUIRED)
find_package(edge_ros_interfaces REQUIRED)
find_package(unitree_api REQUIRED)
find_package(go2_sport_api REQUIRED)
find_package(unitree_sdk2 REQUIRED)

add_executable(real_robot_executor src/executor.cpp)
target_include_directories(real_robot_executor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_executable(robot_state src/robot_state.cpp)
target_link_libraries(robot_state unitree_sdk2)

add_executable(dummy_robot_env src/dummy_robot_env.cpp)

ament_target_dependencies(real_robot_executor
  rclcpp
  geometry_msgs
  unitree_api
  go2_sport_api
)

ament_target_dependencies(dummy_robot_env
  rclcpp
  geometry_msgs
  std_msgs
  edge_msgs
  edge_config
  edge_ros_interfaces
)

install(TARGETS
  real_robot_executor
  robot_state
  dummy_robot_env
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
