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

    this->declare_parameter<double>("roll_limit_deg", 30.0);
    this->declare_parameter<double>("pitch_limit_deg", 30.0);
    this->declare_parameter<double>("height_limit", 0.1);

    int update_rate_hz = this->get_parameter("update_rate").as_int();
    this->get_parameter("print_process_time", print_process_time_);

    // Initialize Safety Checker
    double roll_deg  = this->get_parameter("roll_limit_deg").as_double();
    double pitch_deg = this->get_parameter("pitch_limit_deg").as_double();
    double height = this->get_parameter("height_limit").as_double();
    safety_checker_ = std::make_shared<edge::SafetyChecker>(roll_deg, pitch_deg, height);

    RCLCPP_INFO(this->get_logger(), "Update rate: %d Hz", update_rate_hz);
    RCLCPP_INFO(this->get_logger(), "Print process time: %s", print_process_time_ ? "true" : "false");

    lmi_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(1000 / update_rate_hz),
      std::bind(&PhyTeacherNode::lmiCallback, this)
    );

    sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>("tracking_error", 10,
      std::bind(&PhyTeacherNode::errorCallback, this, std::placeholders::_1)
    );

    pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("phy_teacher_action", 10);

    RCLCPP_INFO(this->get_logger(), "Phy-Teacher Safety Controller Node initialized.");
  }

PhyTeacherNode::~PhyTeacherNode() {}

// Subscription callback to receive tracking error
void PhyTeacherNode::errorCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {

  // Update last error state
  last_error_state_ = msg->data;

  for (int i = 0; i < 10 && i < msg->data.size(); ++i) {
    lmi_optimizer_.tracking_err[i] = msg->data[i];
  }
  got_msg_ = true;
}

void PhyTeacherNode::lmiCallback() {
  // Only proceed if a new message has been received
  if (!got_msg_) return;

  rclcpp::Time start_time;
  if (print_process_time_)
    start_time = this->now();

  bool is_current_safe = safety_checker_->checkSafety(last_error_state_);

  // If teacher is not active
  if (!teacher_active_) {
    // If system is unsafe, activate teacher and set patch center
    if (!is_current_safe) {
      patch_center_ = chi_ * last_error_state_;
      teacher_active_ = true;

      // Update lmi optimizer
      lmi_optimizer_.update();
      RCLCPP_WARN(this->get_logger(), "⚠️ System is UNSAFE! Phy-Teacher is Activated.");
    }
    // If the system is safe, do nothing
    else {
      RCLCPP_INFO(this->get_logger(), "System is SAFE. Phy-Teacher is Inactive.");
    }
    // // If teacher is not active, publish zero gains
    // std_msgs::msg::Float64MultiArray gain_msg;
    // gain_msg.data.resize(72, 0.0);  // 6x6 F_kp and 6x6 F_kd
    // pub_->publish(gain_msg);
  } else {
    if (is_current_safe) {
      teacher_active_ = false;
      RCLCPP_INFO(this->get_logger(), "✅ System returned to SAFE state. Phy-Teacher is Deactivated.");
    } else {
      RCLCPP_INFO(this->get_logger(), "⚠️ Phy-Teacher is Active. System is still UNSAFE.");
    }
  }

  if (teacher_active_) {
    // Update LMI optimizer with current tracking error
    lmi_optimizer_.update();
  } else {
    // If teacher is not active, set gains to zero
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        lmi_optimizer_.F_kp[i][j] = 0.0;
        lmi_optimizer_.F_kd[i][j] = 0.0;
      }
    }
  }


  geometry_msgs::msg::Accel teacher_action;
  teacher_action.linear

  // Generate message
  std_msgs::msg::Float64MultiArray gain_msg;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      gain_msg.data.push_back(lmi_optimizer_.F_kp[i][j]);
      gain_msg.data.push_back(lmi_optimizer_.F_kd[i][j]);
    }
  }

  // Publish message
  pub_->publish(gain_msg);

  if (print_process_time_) {
    rclcpp::Time end_time = this->now();
    double duration_ms = (end_time - start_time).seconds() * 1000.0;
    RCLCPP_INFO(this->get_logger(), "Published control gains. Process time: %.3f ms", duration_ms);
  }

  got_msg_ = false;   // Reset flag until next message is received
}

} // namespace edge


int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<edge::PhyTeacherNode>());
  rclcpp::shutdown();
  return 0;
}
