// phy_teacher_node.cpp

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "phy_teacher_node.hpp"

namespace edge::phy_teacher {
  
PhyTeacherNode::PhyTeacherNode(const rclcpp::NodeOptions & options)
: Node("phy_teacher_node", options) {
    // Declare parameter and get update rate
    this->declare_parameter<int>("update_rate", 30);
    this->declare_parameter<bool>("print_process_time", false);

    this->declare_parameter<double>("roll_limit_deg", 0.0);
    this->declare_parameter<double>("pitch_limit_deg", 0.0);
    this->declare_parameter<double>("height_limit", 0.05);

    int update_rate_hz = this->get_parameter("update_rate").as_int();
    this->get_parameter("print_process_time", print_process_time_);

    // Initialize Safety Checker
    double roll_deg  = this->get_parameter("roll_limit_deg").as_double();
    double pitch_deg = this->get_parameter("pitch_limit_deg").as_double();
    double height = this->get_parameter("height_limit").as_double();
    safety_checker_ = std::make_shared<edge::config::SafetyChecker>(roll_deg, pitch_deg, height);

    RCLCPP_INFO(this->get_logger(), "Update rate: %d Hz", update_rate_hz);
    RCLCPP_INFO(this->get_logger(), "Print process time: %s", print_process_time_ ? "true" : "false");

    lmi_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(1000 / update_rate_hz),
      std::bind(&PhyTeacherNode::lmiCallback, this)
    );

    sub_ = this->create_subscription<edge_msgs::msg::TrackingError>("tracking_error", 10,
      std::bind(&PhyTeacherNode::errorCallback, this, std::placeholders::_1)
    );

    pub_ = this->create_publisher<edge_msgs::msg::Action>("phy_teacher_action", 10);

    RCLCPP_INFO(this->get_logger(), "Phy-Teacher Safety Controller Node initialized.");
  }

PhyTeacherNode::~PhyTeacherNode() {}

// Subscription callback to receive tracking error
void PhyTeacherNode::errorCallback(const edge_msgs::msg::TrackingError::SharedPtr msg) {

  // Update last error state
  last_error_state_ = edge::ros_interfaces::TrackingErrorMsgToEigen(*msg);

  for (int i = 0; i < 10 && i < msg->tracking_error.size(); ++i) {
    lmi_optimizer_.tracking_err[i] = msg->tracking_error[i];
  }

  RCLCPP_INFO(this->get_logger(), "Received tracking error message.");
  got_msg_ = true;
}

void PhyTeacherNode::lmiCallback() {
  // Only proceed if a new message has been received
  if (!got_msg_) return;

  auto err_state_ = edge::ros_interfaces::EigenToVector(last_error_state_);
  bool is_current_safe = safety_checker_->checkSafety(err_state_);

  // If teacher is not active
  if (!teacher_active_) {
    // If system is unsafe, activate teacher and set patch center
    if (!is_current_safe) {
      patch_center_ = chi_ * last_error_state_;
      teacher_active_ = true;

      if (print_process_time_)
        start_time_ = this->now();

      // Update lmi optimizer
      lmi_optimizer_.update();

      if (print_process_time_) {
        end_time_ = this->now();
        double duration_ms = (end_time_ - start_time_).seconds() * 1000.0;
        RCLCPP_INFO(this->get_logger(), "LMI optimization completed. Process time: %.3f ms", duration_ms);
      }

      RCLCPP_WARN(this->get_logger(), "⚠️ System is UNSAFE! Phy-Teacher is Activated.");
    }
    // If the system is safe, do nothing
    else {
      RCLCPP_INFO(this->get_logger(), "System is SAFE. Phy-Teacher is Inactive.");
    }
  } 
  // If teacher is active
  else {

    // Check if system has returned to safe state, deactivate if so
    if (is_current_safe) {
      teacher_active_ = false;
      RCLCPP_INFO(this->get_logger(), "✅ System returned to SAFE state. Phy-Teacher is Deactivated.");
    } else {
      RCLCPP_INFO(this->get_logger(), "⚠️ Phy-Teacher is Active. System is still UNSAFE.");
    }
  }

  // Prepare and publish action message
  edge_msgs::msg::Action teacher_action_msg;
  if (teacher_active_) {
    // Compute the action according to the center of the safety patch
    edge::ros_interfaces::Vector12d error_from_patch = last_error_state_ - patch_center_;
    auto kp_term = lmi_optimizer_.F_kp * error_from_patch.head(6) * 3;
    auto kd_term = lmi_optimizer_.F_kd * error_from_patch.tail(6) * 3;

    auto result = kp_term + kd_term;
    teacher_action_msg = edge::ros_interfaces::EigenToActionMsg(result);
  } else {
    // If teacher is not active, set gains to zero
    teacher_action_msg = edge::ros_interfaces::EigenToActionMsg(edge::ros_interfaces::Vector6d::Zero());
  }

  pub_->publish(teacher_action_msg);

  // if (print_process_time_) {
  //   rclcpp::Time end_time = this->now();
  //   double duration_ms = (end_time - start_time).seconds() * 1000.0;
  //   RCLCPP_INFO(this->get_logger(), "Published control gains. Process time: %.3f ms", duration_ms);
  // }

  got_msg_ = false;   // Reset flag until next message is received
}

} // namespace edge::phy_teacher


int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<edge::phy_teacher::PhyTeacherNode>());
  rclcpp::shutdown();
  return 0;
}
