#include "edge_trigger/trigger.hpp"


using namespace std::chrono_literals;
using std::placeholders::_1;

namespace edge::trigger {

TriggerNode::TriggerNode(const rclcpp::NodeOptions & options)
: Node("trigger_node", options)
{
  RCLCPP_INFO(this->get_logger(), "🔘 TriggerNode initialized.");

  // --- Declare parameters ---
  double roll_deg  = this->declare_parameter<double>("roll_limit_deg", 30.0);
  double pitch_deg = this->declare_parameter<double>("pitch_limit_deg", 30.0);
  double height = this->declare_parameter<double>("height_limit", 0.1);
  update_rate_ = this->declare_parameter<double>("update_rate", 150.0);

  RCLCPP_INFO(this->get_logger(), "🔘 TriggerNode update rate set to %f Hz", update_rate_);
  RCLCPP_INFO(this->get_logger(), "🔘 Initializing SafetyChecker with roll=%.1f°, pitch=%.1f°, height=%.2f",
              roll_deg, pitch_deg, height);

  safety_checker_ = std::make_shared<edge::config::SafetyChecker>(roll_deg, pitch_deg, height);

  // Initialize cached variables
  last_error_state_ = edge::ros_interfaces::TrackingErrorMsgToEigen(edge_msgs::msg::TrackingError());        // initial state
  last_student_action_ = edge::ros_interfaces::ActionMsgToEigen(edge_msgs::msg::Action());     // initial action
  last_teacher_action_ = edge::ros_interfaces::ActionMsgToEigen(edge_msgs::msg::Action());     // initial action
  is_last_state_safe_ = true;      // initial alert state

  // Subscribers (state, teacher action, student action)
  sub_error_ = this->create_subscription<edge_msgs::msg::TrackingError>(
    "tracking_error", 50, std::bind(&TriggerNode::errorCallback, this, _1));

  sub_teacher_ = this->create_subscription<edge_msgs::msg::Action>(
    "phy_teacher_action", 50, std::bind(&TriggerNode::teacherCallback, this, _1));

  sub_student_ = this->create_subscription<edge_msgs::msg::Action>(
    "drl_student_action", 50, std::bind(&TriggerNode::studentCallback, this, _1));

  // Publishers (for alert and terminal action)
  pub_alert_ = this->create_publisher<std_msgs::msg::Bool>("is_safe", 50);
  pub_action_ = this->create_publisher<edge_msgs::msg::Action>("terminal_action", 50);

  auto period = std::chrono::duration<double>(1.0 / update_rate_);
  timer_ = this->create_wall_timer(period, std::bind(&TriggerNode::updateLoop, this));

}

void TriggerNode::studentCallback(const edge_msgs::msg::Action::SharedPtr msg)
{
  last_student_action_ = edge::ros_interfaces::ActionMsgToEigen(*msg);
}

void TriggerNode::teacherCallback(const edge_msgs::msg::Action::SharedPtr msg)
{
  last_teacher_action_ = edge::ros_interfaces::ActionMsgToEigen(*msg);
}

void TriggerNode::errorCallback(const edge_msgs::msg::TrackingError::SharedPtr msg)
{
  last_error_state_ = edge::ros_interfaces::TrackingErrorMsgToEigen(*msg);
    // RCLCPP_INFO(this->get_logger(), "Received new error state: %.3f, %.3f, %.3f",
    //             last_error_state_[0], last_error_state_[1], last_error_state_[2]);
}

// -------------------- Main periodic update --------------------
void TriggerNode::updateLoop()
{
  std::vector<double> v(last_error_state_.data(), last_error_state_.data() + last_error_state_.size());
  bool is_safe = safety_checker_->checkSafety(v);

  RCLCPP_INFO(this->get_logger(), "TriggerNode: is_safe = %s", is_safe ? "true" : "false");
  edge_msgs::msg::Action action_msg;
  std_msgs::msg::Bool alert_msg;
  alert_msg.data = is_safe;

  // Decide which action to publish
  if (is_safe)
    action_msg = edge::ros_interfaces::EigenToActionMsg(last_student_action_);
  else
    action_msg = edge::ros_interfaces::EigenToActionMsg(last_teacher_action_);

  pub_action_->publish(action_msg);
  pub_alert_->publish(alert_msg);

}

}  // namespace edge::trigger


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<edge::trigger::TriggerNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
