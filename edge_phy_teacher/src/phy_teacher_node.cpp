/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include "phy_teacher.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace edge::phy_teacher {
  
PhyTeacher::PhyTeacher(const rclcpp::NodeOptions & options)
: Node("phy_teacher", options) {

    // ------------------------------------------------------
    // Declare ROS parameters with defaults
    // ------------------------------------------------------
    update_rate_        = this->declare_parameter<int>("update_rate", 200);

    // PhyTeacher config
    tau_                 = this->declare_parameter<int>("tau", 20);
    chi_                 = this->declare_parameter<double>("chi", 0.25);
    teacher_enable_      = this->declare_parameter<bool>("teacher_enable", true);
    trigger_type_        = this->declare_parameter<std::string>("trigger_type", "event");
    curr_action_         = edge::ros_interfaces::Vector6d::Zero();
    last_action_         = edge::ros_interfaces::Vector6d::Zero();

    // Initialize Safety Monitor
    safety_monitor_ = std::make_shared<edge::config::SafetyMonitor>();

    RCLCPP_INFO(this->get_logger(), "Update rate: %d Hz", update_rate_);

    // Initialize Default Gains
    std::vector<double> default_kp_raw = this->declare_parameter<std::vector<double>>("DEFAULT_KP", std::vector<double>(36, 0.0));
    std::vector<double> default_kd_raw = this->declare_parameter<std::vector<double>>("DEFAULT_KD", std::vector<double>(36, 0.0));
    set_default_kp(edge::ros_interfaces::VectorToEigen<6, 6>(default_kp_raw));
    set_default_kd(edge::ros_interfaces::VectorToEigen<6, 6>(default_kd_raw));

    // Timer
    control_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(1000 / update_rate_),
      std::bind(&PhyTeacher::controlCallback, this)
    );

    // Subscribers
    state_sub_ = this->create_subscription<edge_msgs::msg::Vec12Msg>("tracking_error", 10,
      std::bind(&PhyTeacher::errorCallback, this, std::placeholders::_1)
    );

    obstacle_sub_ = this->create_subscription<std_msgs::msg::Float32>("/obstacle_min_distance", 10,
      std::bind(&PhyTeacher::obstacleDistanceCallback, this, std::placeholders::_1)
    );

    goal_sub_ = this->create_subscription<std_msgs::msg::Float32>("/goal_distance", 10,
      std::bind(&PhyTeacher::goalDistanceCallback, this, std::placeholders::_1)
    );

    gain_synthesizer_sub_ = this->create_subscription<edge_msgs::msg::ControlGain>("synthesized_gain", 10,
      std::bind(&PhyTeacher::sythesizedGainCallback, this, std::placeholders::_1)
    );

    // Publishers
    gain_synthesizer_pub_ = this->create_publisher<edge_msgs::msg::Vec12Msg>("control_state", 10);
    action_pub_ = this->create_publisher<edge_msgs::msg::EdgeAction>("phy_teacher_action", 10);

    RCLCPP_INFO(this->get_logger(), "Phy-Teacher Safety Patch Node initialized.");
  }

PhyTeacher::~PhyTeacher() {}

// Subscription callback to receive tracking error
void PhyTeacher::errorCallback(const edge_msgs::msg::Vec12Msg::SharedPtr msg) {
    // Update error state
    error_state_ = edge::ros_interfaces::Vec12MsgToEigen(*msg);
    RCLCPP_INFO(this->get_logger(), "Received tracking error message.");
    got_msg_ = true;
}

void PhyTeacher::sythesizedGainCallback(const edge_msgs::msg::ControlGain::SharedPtr msg) {
    opt_status_ = msg->status;
    Eigen::Map<const Eigen::Matrix<double,6,6,Eigen::RowMajor>> kp_map(msg->kp.data());
    Eigen::Map<const Eigen::Matrix<double,6,6,Eigen::RowMajor>> kd_map(msg->kd.data());
    F_kp = kp_map;
    F_kd = kd_map;
}

void PhyTeacher::obstacleDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg) {
    // Update obstacle distances
    obstacle_distance_ = msg->data;
}

void PhyTeacher::goalDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg) {
    // Update goal distances
    goal_distance_ = msg->data;
}

void PhyTeacher::controlCallback() {
    // Only proceed if a new message has been received
    if (!got_msg_) return;

    auto err_state_ = edge::ros_interfaces::EigenToVector(error_state_);
    bool is_current_safe = safety_monitor_->evaluateSafety(edge::config::RuleType::RELATIVE, 
                                                           err_state_, obstacle_distance_, goal_distance_);

    // If teacher is not active
    if (!teacher_active_) {
        // If system is unsafe, activate teacher and set patch center
        if (!is_current_safe) {
            patch_center_ = chi_ * error_state_;
            teacher_active_ = true;
            gain_synthesizer_pub_->publish(edge::ros_interfaces::EigenToVec12Msg(error_state_));
            RCLCPP_WARN(this->get_logger(), "⚠️ System is UNSAFE! Phy-Teacher is Activated.");
        }
        // If the system is safe, do nothing
        else {
            RCLCPP_INFO(this->get_logger(), "✅ System is SAFE. Phy-Teacher is Inactive.");
        }
    } 
    // If teacher is active
    else {
        // Check if system has returned to safe state, deactivate if so
        if (is_current_safe) {
            teacher_active_ = false;
            RCLCPP_INFO(this->get_logger(), "↩️ System returned to SAFE state. Phy-Teacher is Deactivated.");
        } else {
            RCLCPP_INFO(this->get_logger(), "⚠️ Phy-Teacher is Active. System is still UNSAFE.");
        }
    }

    // Prepare and publish action message
    edge_msgs::msg::EdgeAction teacher_action_msg;
    edge::utils::ActionMode teacher_mode = edge::utils::ActionMode::TEACHER;

    // Compute control gains if teacher is active
    if (teacher_active_) {
        // Compute the action according to the center of the safety patch
        edge::ros_interfaces::Vector12d error_from_patch = error_state_ - patch_center_;
        if (opt_status_ == 0) {
            RCLCPP_INFO(this->get_logger(), "LMI optimization succeeded! Using optimized gains.");
            // auto kp_term = F_kp * error_from_patch.head(6);
            // auto kd_term = F_kd * error_from_patch.tail(6);
            auto kp_term = default_F_kp * error_from_patch.head(6);
            auto kd_term = default_F_kd * error_from_patch.tail(6);
            curr_action_ = kp_term + kd_term;
        } else {
            RCLCPP_ERROR(this->get_logger(), "LMI optimization failed! Using default gains.");
            auto kp_term = default_F_kp * error_from_patch.head(6);
            auto kd_term = default_F_kd * error_from_patch.tail(6);
            curr_action_ = kp_term + kd_term;
        }
    } else {
        // If teacher is not active, maintain last action
        curr_action_ = last_action_;
    }

    teacher_action_msg = edge::ros_interfaces::EigenToEdgeActionMsg(
        curr_action_,  edge_msgs::msg::EdgeAction::BASE_ACCEL, edge::utils::ToMsg(teacher_mode));
    action_pub_->publish(teacher_action_msg);

    got_msg_ = false;   // Reset flag until next message is received
    last_action_ = curr_action_;        // Update last action
}

} // namespace edge::phy_teacher


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<edge::phy_teacher::PhyTeacher>());
    rclcpp::shutdown();
    return 0;
}
