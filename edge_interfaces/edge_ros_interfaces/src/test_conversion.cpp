#include "rclcpp/rclcpp.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"
#include "edge_msgs/msg/action.hpp"
#include "edge_msgs/msg/tracking_error.hpp"

using namespace edge::ros_interfaces;

class MsgConversionTestNode : public rclcpp::Node {
public:
  MsgConversionTestNode() : Node("test_conversion") {
    edge_msgs::msg::Action msg;
    msg.action = {1.0, 2.0, 3.0};
    auto v = ActionMsgToEigen(msg);
    RCLCPP_INFO(this->get_logger(), "Eigen vector: [%f, %f, %f]", v[0], v[1], v[2]);
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MsgConversionTestNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
}
