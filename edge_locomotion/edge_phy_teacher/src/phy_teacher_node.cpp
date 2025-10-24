// phy_teacher_node.cpp

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "lmi_optimizer.hpp"



int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PhyTeacherNode>());
  rclcpp::shutdown();
  return 0;
}
