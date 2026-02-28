/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "msg_utils.hpp"

#include "edge_msgs/msg/vec12_msg.hpp"
#include "edge_msgs/msg/edge_action.hpp"
#include "edge_msgs/msg/edge_vel_action.hpp"
#include "edge_config/safety_monitor.hpp"
#include "edge_msgs/srv/get_switching_stats.hpp"
#include "edge_msgs/srv/reset_switching_stats.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"

#define MAX_VAL 100.0

namespace edge::trigger {

class Trigger : public rclcpp::Node
{
public:
    explicit Trigger(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
    
    void updateLoop();     // Main periodic update loop
    void updateLoopVel();  // Main periodic update loop for velocity action

    bool checkSafety();    // Check safety for switching
    void resetStats();     // Reset switching statistics
    void updateStats(edge::utils::ActionMode);      // Update switching statistics
    void resetCachedVariables(); // Reset cached variables
    
    // ROS2 Callbacks
    void studentCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg);
    void teacherCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg);
    void studentVelCallback(const edge_msgs::msg::EdgeVelAction::SharedPtr msg);
    void teacherVelCallback(const edge_msgs::msg::EdgeVelAction::SharedPtr msg);
    void errorStateCallback(const edge_msgs::msg::Vec12Msg::SharedPtr msg);
    void robotStateCallback(const edge_msgs::msg::Vec12Msg::SharedPtr msg);
    void goalDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg);
    void obstacleDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg);
    void getStatsCallback(
        const std::shared_ptr<edge_msgs::srv::GetSwitchingStats::Request> request,
        std::shared_ptr<edge_msgs::srv::GetSwitchingStats::Response> response
    );
    void resetStatsCallback(
        const std::shared_ptr<edge_msgs::srv::ResetSwitchingStats::Request> request,
        std::shared_ptr<edge_msgs::srv::ResetSwitchingStats::Response> response
    );

    // Cached variables
    double goal_distance_;                                // last received goal distance
    double obstacle_distance_;                            // last received distance to obstacle
    bool is_last_state_safe_;                             // last alert state (is safe or not)
    bool is_last_state_fall_;                             // last fall state (is fall or not)
    edge::ros_interfaces::Vector6d   last_stu_action_;    // last received student action
    edge::ros_interfaces::Vector6d   last_tea_action_;    // last received teacher action
    edge::ros_interfaces::Vector3d   last_stu_vel_action_;    // last received student velocity action
    edge::ros_interfaces::Vector3d   last_tea_vel_action_;    // last received teacher velocity action
    edge::ros_interfaces::Vector12d  last_error_state_;   // last received error state
    edge::ros_interfaces::Vector12d  last_robot_state_;   // last received robot state

    // Statistical tracking
    double student_duration_;                              // Student Mode Accumulated Duration (s)
    double teacher_duration_;                              // Teacher Mode Accumulated Duration (s)
    uint32_t student_action_count_;                        // Number of actions in Student Mode
    uint32_t teacher_action_count_;                        // Number of actions in Teacher Mode
    uint32_t student_switch_count_;                        // Number of switches to Student Mode
    uint32_t teacher_switch_count_;                        // Number of switches to Teacher Mode
    rclcpp::Time mode_start_time_;                         // Mode start time
    edge::utils::ActionMode curr_mode_;                    // Current mode
    edge::utils::ActionMode last_mode_;                    // Last mode
    std::map<std::string, uint32_t> switching_reasons_;    // Switching reasons statistics

    // ROS subscribers
    rclcpp::Subscription<edge_msgs::msg::EdgeAction>::SharedPtr sub_student_;
    rclcpp::Subscription<edge_msgs::msg::EdgeAction>::SharedPtr sub_teacher_;
    rclcpp::Subscription<edge_msgs::msg::EdgeVelAction>::SharedPtr sub_student_vel_;
    rclcpp::Subscription<edge_msgs::msg::EdgeVelAction>::SharedPtr sub_teacher_vel_;
    rclcpp::Subscription<edge_msgs::msg::Vec12Msg>::SharedPtr sub_error_;
    rclcpp::Subscription<edge_msgs::msg::Vec12Msg>::SharedPtr sub_robot_state_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_obstacle_distance_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_goal_distance_;

    // ROS publishers
    rclcpp::Publisher<edge_msgs::msg::EdgeAction>::SharedPtr pub_action_;
    rclcpp::Publisher<edge_msgs::msg::EdgeVelAction>::SharedPtr pub_vel_action_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_alert_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_fall_;

    // ROS services
    rclcpp::Service<edge_msgs::srv::GetSwitchingStats>::SharedPtr srv_get_switching_stats_;
    rclcpp::Service<edge_msgs::srv::ResetSwitchingStats>::SharedPtr srv_reset_switching_stats_;

    double update_rate_;                            // Update rate (Hz)
    int trigger_source_;                            // trigger source: -1 -- none, 0 -- student, 1 -- teacher
    bool print_processing_;                         // whether to print processing info
    bool is_start_with_teacher_;                    // whether to start with teacher mode
    bool is_robot_state_initialized_ = false;       // whether robot state has been initialized
    std::string student_topic_name_;                      // topic name for student action
    std::string teacher_topic_name_;                      // topic name for teacher action
    std::string terminal_action_name_;                   // topic name for publishing terminal action


    rclcpp::TimerBase::SharedPtr timer_;    // Periodic timer
    std::shared_ptr<edge::config::SafetyMonitor> safety_monitor_;       // Safety monitor
};

}  // namespace edge::trigger