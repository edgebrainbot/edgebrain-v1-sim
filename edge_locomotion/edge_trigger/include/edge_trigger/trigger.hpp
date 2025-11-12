#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "msg_utils.hpp"
#include "edge_config/safety_monitor.hpp"
#include "edge_msgs/msg/edge_action.hpp"
#include "edge_msgs/msg/tracking_error.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"


namespace edge::trigger {

class Trigger : public rclcpp::Node
{
public:
  explicit Trigger(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void studentCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg);
  void teacherCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg);
  void errorCallback(const edge_msgs::msg::TrackingError::SharedPtr msg);
  bool checkSafety();
  void updateLoop();

  // Cached variables
  edge::ros_interfaces::Vector12d  last_error_state_;         // last received error state
  edge::ros_interfaces::Vector6d   last_student_action_;      // last received student action
  edge::ros_interfaces::Vector6d   last_teacher_action_;      // last received teacher action
  bool is_last_state_safe_;        // last alert state (is safe)
  bool print_processing_;         // whether to print processing info

  // Safety-conditions
  double update_rate_;

  std::shared_ptr<edge::config::SafetyMonitor> safety_monitor_;

  // ROS Interfaces
  rclcpp::Subscription<edge_msgs::msg::EdgeAction>::SharedPtr sub_student_;
  rclcpp::Subscription<edge_msgs::msg::EdgeAction>::SharedPtr sub_teacher_;
  rclcpp::Subscription<edge_msgs::msg::TrackingError>::SharedPtr sub_error_;
  rclcpp::Publisher<edge_msgs::msg::EdgeAction>::SharedPtr pub_action_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_alert_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace edge