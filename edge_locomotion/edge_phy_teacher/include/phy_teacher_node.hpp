#ifndef PHY_TEACHER_NODE_HPP
#define PHY_TEACHER_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/float64_multi_array.hpp"
#include "edge_msgs/msg/tracking_error.hpp"
#include "edge_msgs/msg/action.hpp"
#include "lmi_optimizer.hpp"
#include "edge_config/safety_checker.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"

namespace edge::phy_teacher {

class PhyTeacherNode : public rclcpp::Node {
public:

  explicit PhyTeacherNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~PhyTeacherNode() override;

private:
  void errorCallback(const edge_msgs::msg::TrackingError::SharedPtr msg);
  void lmiCallback();

  // For phy-teacher logistics
  int tau_ = 20;                          // Time steps for safety patch
  double chi_ = 0.25;                       // Hyperparameter χ
  bool teacher_enable_ = true;              // Whether the teacher is enabled
  bool teaching_to_learn_ = true;      // Whether to teach to learn
  std::string trigger_type_ = "event";     // Trigger type: "event" or "periodic"

  int update_rate_;                   // Update rate in Hz
  bool got_msg_ = false;
  bool teacher_active_ = false;            // Whether the teacher is active
  bool print_process_time_ = false;
  edge::ros_interfaces::Vector12d  error_state_;          // received error state
  edge::ros_interfaces::Vector12d  patch_center_;         // Center of the safety patch

  lmi::LMIOptimizer lmi_optimizer_;

  // For profiling
  rclcpp::Time start_time_;   
  rclcpp::Time end_time_;

  // Safety-conditions
  double roll_limit_rad_;
  double pitch_limit_rad_;
  double height_limit_;
  bool is_last_safe_ = true;
  std::shared_ptr<edge::config::SafetyChecker> safety_checker_;

  rclcpp::Subscription<edge_msgs::msg::TrackingError>::SharedPtr sub_;
  rclcpp::Publisher<edge_msgs::msg::Action>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr lmi_timer_;
};

}  // namespace edge::phy_teacher

#endif  // PHY_TEACHER_NODE_HPP