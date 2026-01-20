/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

 #include <memory>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include "lmi_optimizer.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

class ControlGainListener : public rclcpp::Node {
public:
  ControlGainListener()
  : Node("gain_listener")
  {
    subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "control_gain", 10,
      std::bind(&ControlGainListener::on_gain_update, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "ControlGainListener initialized.");
  }

private:
  void on_gain_update(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {

    if (!msg) {
      RCLCPP_ERROR(this->get_logger(), "Received null message in on_gain_update.");
      return;
    }

    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        F_kp(i, j)  = msg->data[i * 6 + j];
        F_kd(i, j)  = msg->data[i * 6 + j + 36];
      }
    }

    lmi::LMIOptimizer::print_matrix6x6(F_kp, "F_kp");
    lmi::LMIOptimizer::print_matrix6x6(F_kd, "F_kd");
  }

  lmi::Mat6x6 F_kp;
  lmi::Mat6x6 F_kd;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ControlGainListener>());
  rclcpp::shutdown();
  return 0;
}
