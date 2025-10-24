#pragma once

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cmath>
#include <vector>
#include <string>

namespace edge {

class TriggerNode : public rclcpp::Node
{
public:
  explicit TriggerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void studentCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void teacherCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void errorCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  bool checkSafety();
  void updateLoop();

  // Cached variables
  std::vector<double> last_error_state_;         // last received error state
  std::vector<double> last_student_action_;      // last received student action
  std::vector<double> last_teacher_action_;      // last received teacher action
  bool is_last_state_safe_;        // last alert state (is safe)

  // Safety-conditions
  double roll_limit_rad_;
  double pitch_limit_rad_;
  double height_limit_;
  double update_rate_hz_;

  // ROS Interfaces
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_student_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_teacher_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_error_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_action_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_alert_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace edge
