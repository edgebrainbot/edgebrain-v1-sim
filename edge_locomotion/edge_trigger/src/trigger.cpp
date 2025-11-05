#include "edge_trigger/trigger.hpp"


using namespace std::chrono_literals;
using std::placeholders::_1;

namespace edge::trigger {

TriggerNode::TriggerNode(const rclcpp::NodeOptions & options)
: Node("trigger_node", options)
{
  RCLCPP_INFO(this->get_logger(), "🔘 TriggerNode initialized.");

  // --- Declare parameters ---
  this->declare_parameter<double>("update_rate", 50.0);
  this->declare_parameter<double>("roll_limit_deg", 30.0);
  this->declare_parameter<double>("pitch_limit_deg", 30.0);
  this->declare_parameter<double>("height_limit", 0.1);

  double roll_deg  = this->get_parameter("roll_limit_deg").as_double();
  double pitch_deg = this->get_parameter("pitch_limit_deg").as_double();
  double height = this->get_parameter("height_limit").as_double();
  update_rate_hz_ = this->get_parameter("update_rate").as_double();

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
    "tracking_error", 10, std::bind(&TriggerNode::errorCallback, this, _1));

  sub_teacher_ = this->create_subscription<edge_msgs::msg::Action>(
    "phy_teacher_action", 10, std::bind(&TriggerNode::teacherCallback, this, _1));

  sub_student_ = this->create_subscription<edge_msgs::msg::Action>(
    "drl_student_action", 10, std::bind(&TriggerNode::studentCallback, this, _1));

  // Publishers (for alert and terminal action)
  pub_alert_ = this->create_publisher<std_msgs::msg::Bool>("is_safe", 10);
  pub_action_ = this->create_publisher<edge_msgs::msg::Action>("terminal_action", 10);

  auto period = std::chrono::duration<double>(1.0 / update_rate_hz_);
  timer_ = this->create_wall_timer(period, std::bind(&TriggerNode::updateLoop, this));

  // RCLCPP_INFO(this->get_logger(), "✅ Parameters loaded: roll=%.1f°, pitch=%.1f°, height=%.2f",
  //             roll_limit_rad_ * 180.0 / M_PI, pitch_limit_rad_ * 180.0 / M_PI,
  //             height_limit_);
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

// -------------------- Safety Check --------------------
// bool TriggerNode::checkSafety()
// {
//   bool is_safe = true;
//   std::vector<std::string> reasons;

//   double height = last_error_state_[0];    
//   double roll = last_error_state_[1];
//   double pitch = last_error_state_[2];
  
//   // Check roll
//   if (std::abs(roll) > roll_limit_rad_) {
//     is_safe = false;
//     reasons.push_back("roll = " + std::to_string(roll * 180.0 / M_PI) + "°");
//   }

//   // Check pitch
//   if (std::abs(pitch) > pitch_limit_rad_) {
//     is_safe = false;
//     reasons.push_back("pitch = " + std::to_string(pitch * 180.0 / M_PI) + "°");
//   }

//   // Check height
//   if (std::abs(height) > height_limit_) {
//     is_safe = false;
//     reasons.push_back("height = " + std::to_string(height) + "m");
//   }

//   // Logging
//   if (!is_safe) {
//     std::string msg = "⚠️ Current System is UNSAFE! Reasons: ";
//     for (auto &r : reasons) msg += r + " | ";

//     if (is_last_state_safe_)
//         msg += "System leaving the learning space! Safety Triggered.";
//     else
//         msg += "System still outside the learning space.";

//     RCLCPP_WARN(this->get_logger(), "%s", msg.c_str());
//   } else {
//     if (!is_last_state_safe_)
//       RCLCPP_INFO(this->get_logger(), "✅ System returned to the learning space.");
//     else
//       RCLCPP_INFO(this->get_logger(), "✅ System remains within the learning space.");
//   }

//   is_last_state_safe_ = is_safe;    // update last state
  
//   return is_safe;
// }

}  // namespace edge::trigger


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<edge::trigger::TriggerNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
