#ifndef PHY_TEACHER_NODE_HPP
#define PHY_TEACHER_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/float64_multi_array.hpp"
#include "lmi_optimizer.hpp"

class PhyTeacherNode : public rclcpp::Node {
public:
  PhyTeacherNode() : Node("phy_teacher_node") {
    // Declare parameter and get update rate
    this->declare_parameter<int>("update_rate", 30);
    this->declare_parameter<bool>("print_process_time", false);

    int update_rate_hz = this->get_parameter("update_rate").as_int();
    this->get_parameter("print_process_time", print_process_time_);

    RCLCPP_INFO(this->get_logger(), "Update rate: %d Hz", update_rate_hz);
    RCLCPP_INFO(this->get_logger(), "Print process time: %s", print_process_time_ ? "true" : "false");

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(1000 / update_rate_hz),
      std::bind(&PhyTeacherNode::timer_callback, this)
    );

    sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "tracking_error", update_rate_hz,
      std::bind(&PhyTeacherNode::data_callback, this, std::placeholders::_1)
    );

    pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
      "control_gain", update_rate_hz
    );

    RCLCPP_INFO(this->get_logger(), "Phy-Teacher Safety Controller Node initialized.");
  }

private:
  void data_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    for (int i = 0; i < 10 && i < msg->data.size(); ++i) {
      safety_controller_.tracking_err[i] = msg->data[i];
    }
    got_msg_ = true;
  }

  void timer_callback() {
  if (!got_msg_) return;

  rclcpp::Time start_time;
  if (print_process_time_)
    start_time = this->now();

  // Update controller
  lmi_optimizer_.update();

  // Generate message
  std_msgs::msg::Float64MultiArray gain_msg;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      gain_msg.data.push_back(safety_controller_.F_kp[i][j]);
      gain_msg.data.push_back(safety_controller_.F_kd[i][j]);
    }
  }

  // Publish message
  pub_->publish(gain_msg);

  if (print_process_time_) {
    rclcpp::Time end_time = this->now();
    double duration_ms = (end_time - start_time).seconds() * 1000.0;
    RCLCPP_INFO(this->get_logger(), "Published control gains. Process time: %.3f ms", duration_ms);
    }
  }

  std::vector<double> patch_center_;
  bool got_msg_ = false;
  bool print_process_time_ = false;
  lmi::LMIOptimizer lmi_optimizer_;

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

#endif  // PHY_TEACHER_NODE_HPP