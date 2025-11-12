cmake_minimum_required(VERSION 3.8)
project(edge_phy_teacher LANGUAGES C CXX)

# ------------------ Compiler flags ------------------
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# ------------------ Dependencies ------------------
find_package(rclcpp REQUIRED)
find_package(BLAS REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(LAPACK REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(edge_msgs REQUIRED)
find_package(edge_config REQUIRED)
find_package(edge_ros_interfaces REQUIRED)

# ------------------ ecvxcone C library ------------------
set(ecvxcone_headers
  include/ecvxcone/cvxopt.h
  include/ecvxcone/lapack.h
  include/ecvxcone/misc.h
  include/ecvxcone/blas.h
  include/ecvxcone/base.h
  include/ecvxcone/solver.h
  include/ecvxcone/cpg_workspace.h
  include/ecvxcone/cpg_solve.h
)

set(ecvxcone_sources
  src/ecvxcone/base.c
  src/ecvxcone/blas.c
  src/ecvxcone/debug.c
  src/ecvxcone/dense.c
  src/ecvxcone/lapack.c
  src/ecvxcone/misc.c
  src/ecvxcone/solver.c
  src/ecvxcone/sparse.c
  src/ecvxcone/solver_utils.c
  src/ecvxcone/cpg_solve.c
  src/ecvxcone/cpg_workspace.c
)

add_library(ecvxcone SHARED ${ecvxcone_sources})
target_include_directories(ecvxcone PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/ecvxcone)
set_target_properties(ecvxcone PROPERTIES
  LINKER_LANGUAGE C
  POSITION_INDEPENDENT_CODE ON
)

# ------------------ C++ library ------------------
add_library(lmi_optimizer_lib
  src/lmi_optimizer.cpp
)
target_include_directories(lmi_optimizer_lib PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(lmi_optimizer_lib
  ecvxcone
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
)
ament_target_dependencies(lmi_optimizer_lib
  rclcpp
  std_msgs
)

# ------------------ Executable ------------------
add_executable(phy_teacher src/phy_teacher.cpp)
target_include_directories(phy_teacher PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(phy_teacher lmi_optimizer_lib)

# Ensure RPATH includes relative lib/
set_target_properties(phy_teacher PROPERTIES
  INSTALL_RPATH "$ORIGIN/../lib"
)
ament_target_dependencies(phy_teacher rclcpp edge_msgs edge_ros_interfaces edge_config)

add_executable(mock_publisher test/mock_publisher.cpp)
ament_target_dependencies( 
  mock_publisher 
  rclcpp 
  std_msgs
  edge_config
  edge_msgs
)

add_executable(control_gain_listener test/control_gain_listener.cpp)
target_link_libraries(control_gain_listener lmi_optimizer_lib)
ament_target_dependencies(control_gain_listener rclcpp)
target_include_directories(control_gain_listener PUBLIC include)


# ------------------ Install ------------------
install(TARGETS
  phy_teacher
  lmi_optimizer_lib
  ecvxcone
  mock_publisher
  control_gain_listener
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# ------------------ Testing (Optional) ------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()