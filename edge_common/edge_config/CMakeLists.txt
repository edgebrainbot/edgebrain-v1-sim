cmake_minimum_required(VERSION 3.8)
project(edge_config)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml-cpp REQUIRED) 
find_package(edge_msgs REQUIRED)
find_package(edge_ros_interfaces REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/safety_monitor.cpp
  src/reward_composer.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}
  yaml-cpp
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  std_msgs
  edge_msgs
  edge_ros_interfaces
)

add_executable(virtual_safety_indicator_node src/virtual_safety_indicator_node.cpp)
ament_target_dependencies(virtual_safety_indicator_node
  rclcpp std_msgs visualization_msgs
)

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS virtual_safety_indicator_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(rclcpp std_msgs)

ament_package()
