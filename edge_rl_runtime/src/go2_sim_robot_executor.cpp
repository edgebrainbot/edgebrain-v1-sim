/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <functional>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

class Go2SimRobotExecutorNode : public rclcpp::Node
{
public:
  Go2SimRobotExecutorNode() : Node("go2_sim_robot_executor")
  {
    cmd_vel_topic_ = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");

    request_pub_ = create_publisher<unitree_api::msg::Request>(request_topic_, 10);
    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      cmd_vel_topic_, 20,
      std::bind(&Go2SimRobotExecutorNode::cmdVelCallback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(get_logger(), "Listening %s -> publishing %s", cmd_vel_topic_.c_str(), request_topic_.c_str());
  }

private:
  void cmdVelCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
  {
    std::cout << "Received cmd_vel: linear.x=" << msg->twist.linear.x
              << ", linear.y=" << msg->twist.linear.y
              << ", angular.z=" << msg->twist.angular.z << std::endl;

    const double vx = msg->twist.linear.x;
    const double vy = msg->twist.linear.y;
    const double wz = msg->twist.angular.z;

    if (std::fabs(vx) < stop_epsilon_ && std::fabs(vy) < stop_epsilon_ && std::fabs(wz) < stop_epsilon_) {
      sport_client_.StopMove(request_);
    } else {
      sport_client_.Move(request_, vx, vy, wz);
    }
    request_pub_->publish(request_);
  }

  std::string cmd_vel_topic_;
  std::string request_topic_;
  double stop_epsilon_;

  unitree_api::msg::Request request_;
  SportClient sport_client_;
  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr request_pub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go2SimRobotExecutorNode>());
  rclcpp::shutdown();
  return 0;
}
