#include "edge_trigger/trigger.hpp"

using std::placeholders::_1;
using std::placeholders::_2;
using namespace std::chrono_literals;

namespace edge::trigger {

Trigger::Trigger(const rclcpp::NodeOptions & options)
: Node("trigger", options)
{
    // Load parameters
    update_rate_ = this->declare_parameter<double>("update_rate", 150.0);
    print_processing_ = this->declare_parameter<bool>("print_processing", false);
    is_start_with_teacher_ = this->declare_parameter<bool>("start_with_teacher", true);

    RCLCPP_INFO(this->get_logger(), "🔘 Trigger update rate is set to %.1f Hz", update_rate_);
    RCLCPP_INFO(this->get_logger(), "🔘 Trigger print processing is set to %s", print_processing_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "🔘 Trigger start with teacher mode is set to %s", is_start_with_teacher_ ? "true" : "false");
    
    // Initialize statistical tracking variables
    resetStats();

    // Initialize cached variables
    resetCachedVariables();

    // Subscribers (state, teacher action, student action, obstacle distance)
    sub_error_ = this->create_subscription<edge_msgs::msg::Vec12Msg>(
        "tracking_error", 50, std::bind(&Trigger::errorStateCallback, this, _1));

    sub_robot_state_ = this->create_subscription<edge_msgs::msg::Vec12Msg>(
        "robot_state", 50, std::bind(&Trigger::robotStateCallback, this, _1));

    sub_teacher_ = this->create_subscription<edge_msgs::msg::EdgeAction>(
        "phy_teacher_action", 50, std::bind(&Trigger::teacherCallback, this, _1));

    sub_student_ = this->create_subscription<edge_msgs::msg::EdgeAction>(
        "drl_student_action", 50, std::bind(&Trigger::studentCallback, this, _1));

    sub_obstacle_distance_ = this->create_subscription<std_msgs::msg::Float32>(
        "/obstacle_min_distance", 50, std::bind(&Trigger::obstacleDistanceCallback, this, _1));

    sub_goal_distance_ = this->create_subscription<std_msgs::msg::Float32>(
        "/goal_distance", 50, std::bind(&Trigger::goalDistanceCallback, this, _1));

    // Publishers (for alert and terminal action)
    pub_alert_  = this->create_publisher<std_msgs::msg::Bool>("is_safe", 50);
    pub_fall_   = this->create_publisher<std_msgs::msg::Bool>("is_fall", 50);
    pub_action_ = this->create_publisher<edge_msgs::msg::EdgeAction>("terminal_action", 50);

    // Service for getting switching statistics
    srv_get_switching_stats_ = this->create_service<edge_msgs::srv::GetSwitchingStats>(
        "get_switching_stats",
        std::bind(&Trigger::getStatsCallback, this, _1, _2));

    // Service for resetting switching statistics
    srv_reset_switching_stats_ = this->create_service<edge_msgs::srv::ResetSwitchingStats>(
        "reset_switching_stats",
        std::bind(&Trigger::resetStatsCallback, this, _1, _2));

    // Timer for periodic update
    auto period = std::chrono::duration<double>(1.0 / update_rate_);
    timer_ = this->create_wall_timer(period, std::bind(&Trigger::updateLoop, this));

    // Initialize Safety Monitor
    safety_monitor_ = std::make_shared<edge::config::SafetyMonitor>();

    RCLCPP_INFO(this->get_logger(), "🔘 The Trigger Component is initialized.");

}

void Trigger::studentCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg)
{
    last_stu_action_ = edge::ros_interfaces::EdgeActionMsgToEigen(*msg);
}

void Trigger::teacherCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg)
{
    last_tea_action_ = edge::ros_interfaces::EdgeActionMsgToEigen(*msg);
}

void Trigger::errorStateCallback(const edge_msgs::msg::Vec12Msg::SharedPtr msg)
{
    last_error_state_ = edge::ros_interfaces::Vec12MsgToEigen(*msg);
    // RCLCPP_INFO(this->get_logger(), "Received new error state: %.3f, %.3f, %.3f",
    //             last_error_state_[0], last_error_state_[1], last_error_state_[2]);
}

void Trigger::robotStateCallback(const edge_msgs::msg::Vec12Msg::SharedPtr msg)
{
    last_robot_state_ = edge::ros_interfaces::Vec12MsgToEigen(*msg);
    if (!is_robot_state_initialized_) is_robot_state_initialized_ = true;
}

void Trigger::obstacleDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg)
{
    // Placeholder for distance callback
    obstacle_distance_ = msg->data;
}

void Trigger::goalDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg)
{
    // Placeholder for distance callback
    goal_distance_ = msg->data;
}

void Trigger::resetStatsCallback(
    const std::shared_ptr<edge_msgs::srv::ResetSwitchingStats::Request> /*request*/,
    std::shared_ptr<edge_msgs::srv::ResetSwitchingStats::Response> response)
{
    try {
        resetStats();   // Reset the switching statistics

        response->success = true;
        response->message = "The Trigger's switching statistics have been reset successfully";
    }
    catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to reset switching statistics: %s", e.what());
        response->success = false;
        response->message = std::string("Failed to reset switching statistics: ") + e.what();
    }
    catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Failed to reset switching statistics: unknown error");
        response->success = false;
        response->message = "Failed to reset switching statistics: unknown error";
    }
}

void Trigger::getStatsCallback(
    const std::shared_ptr<edge_msgs::srv::GetSwitchingStats::Request> /*request*/,
    std::shared_ptr<edge_msgs::srv::GetSwitchingStats::Response> response)
{
    // Fill in switching reasons
    for (const auto& [reason, count] : switching_reasons_) {
        response->switching_reasons.push_back(reason);
        response->reason_counts.push_back(count);
    }

    // Switching frequency and ratios
    uint32_t total_switches = student_switch_count_ + teacher_switch_count_;
    auto student_switch_ratio = total_switches > 0 ? static_cast<float>(student_switch_count_) / total_switches : 0.0f;
    response->total_switches = total_switches;
    response->student_switch_ratio = student_switch_ratio;

    // Action counts and ratios
    uint32_t total_action_count = student_action_count_ + teacher_action_count_;
    auto student_action_ratio = total_action_count > 0 ? static_cast<float>(student_action_count_) / total_action_count : 0.0f;
    response->total_action_count = total_action_count;
    response->student_action_ratio = student_action_ratio;

    // Duration and time ratios
    double total_duration = student_duration_ + teacher_duration_;
    auto student_time_ratio = total_duration > 0.0 ? static_cast<float>(student_duration_) / total_duration : 0.0f;
    response->total_duration = total_duration;
    response->student_time_ratio = student_time_ratio;

    // Print statistics to console
    RCLCPP_INFO(this->get_logger(), "📊 Statistics Query:");
    RCLCPP_INFO(this->get_logger(), "Total Switches: %u", total_switches);
    RCLCPP_INFO(this->get_logger(), "   - Switches to Student Mode: %u", student_switch_count_);
    RCLCPP_INFO(this->get_logger(), "   - Switches to Teacher Mode: %u", teacher_switch_count_);
    RCLCPP_INFO(this->get_logger(), "   - Student Switch Ratio: %.2f", student_switch_ratio);
    RCLCPP_INFO(this->get_logger(), "Total Actions: %u", total_action_count);
    RCLCPP_INFO(this->get_logger(), "   - Actions in Student Mode: %u", student_action_count_);
    RCLCPP_INFO(this->get_logger(), "   - Actions in Teacher Mode: %u", teacher_action_count_);
    RCLCPP_INFO(this->get_logger(), "   - Student Action Ratio: %.2f", student_action_ratio);
    RCLCPP_INFO(this->get_logger(), "Total Duration: %.2f seconds", total_duration);
    RCLCPP_INFO(this->get_logger(), "   - Time in Student Mode: %.2f seconds", student_duration_);
    RCLCPP_INFO(this->get_logger(), "   - Time in Teacher Mode: %.2f seconds", teacher_duration_);
    RCLCPP_INFO(this->get_logger(), "   - Student Time Ratio: %.2f", student_time_ratio);
    
    // Print switching reasons statistics
    if (!switching_reasons_.empty()) {
        RCLCPP_INFO(this->get_logger(), "   Switching Reasons:");
        for (const auto& [reason, count] : switching_reasons_) {
            RCLCPP_INFO(this->get_logger(), "      - %s: %u times", reason.c_str(), count);
        }
    } else {
        RCLCPP_INFO(this->get_logger(), "   No mode switches recorded yet.");
    }
}

void Trigger::resetCachedVariables()
{
    goal_distance_ = is_start_with_teacher_ ? 0.0 : MAX_VAL;
    obstacle_distance_ = is_start_with_teacher_ ? 0.0 : MAX_VAL;
    is_last_state_fall_ = false;     // initial fall state
    is_last_state_safe_ = !is_start_with_teacher_;      // initial alert state
    last_error_state_ = edge::ros_interfaces::Vec12MsgToEigen(edge_msgs::msg::Vec12Msg());            // initial error state
    last_robot_state_ = edge::ros_interfaces::Vec12MsgToEigen(edge_msgs::msg::Vec12Msg());            // initial robot state
    last_stu_action_  = edge::ros_interfaces::EdgeActionMsgToEigen(edge_msgs::msg::EdgeAction());     // initial student action
    last_tea_action_  = edge::ros_interfaces::EdgeActionMsgToEigen(edge_msgs::msg::EdgeAction());     // initial teacher action
}

void Trigger::resetStats()
{
    switching_reasons_.clear();

    mode_start_time_ = this->now();
    
    student_switch_count_ = 0;
    teacher_switch_count_ = 0;

    student_action_count_ = 0;
    teacher_action_count_ = 0;

    student_duration_ = 0.0;
    teacher_duration_ = 0.0;

    curr_mode_ = is_start_with_teacher_ ? edge::utils::ActionMode::TEACHER 
                                        : edge::utils::ActionMode::STUDENT;
    last_mode_ = is_start_with_teacher_ ? edge::utils::ActionMode::TEACHER 
                                        : edge::utils::ActionMode::STUDENT;
}

void Trigger::updateStats(edge::utils::ActionMode terminal_mode)
{
    auto current_time = this->now();
    auto duration = (current_time - mode_start_time_).seconds();

    // Accumulate duration and action count for current mode
    if (curr_mode_ == edge::utils::ActionMode::STUDENT) {
        student_duration_ += duration;
        ++student_action_count_;
    } else {
        teacher_duration_ += duration;
        ++teacher_action_count_;
    }

    // Check for mode switch
    if (terminal_mode != curr_mode_) {
        
        // Record switching reasons (from SafetyMonitor)
        if (terminal_mode == edge::utils::ActionMode::TEACHER) {
            // Student -> Teacher: Record violation reasons
            const auto& reasons = safety_monitor_->getViolationReasons();
            for (const auto& reason : reasons) {
                ++switching_reasons_[reason];
            }
            
            if (print_processing_) {
                RCLCPP_WARN(this->get_logger(), "🔄 Mode Switch: STUDENT -> TEACHER (unsafe)");
            }
            ++teacher_switch_count_;
        } else {
            // Teacher -> Student: Return to safe
            if (print_processing_) {
                RCLCPP_INFO(this->get_logger(), "🔄 Mode Switch: TEACHER -> STUDENT (safe)");
            }
            ++student_switch_count_;
        }
        
        // Update mode and reset timer
        last_mode_ = curr_mode_;
        curr_mode_ = terminal_mode;
        mode_start_time_ = current_time;
    }
}


// -------------------- Main periodic update --------------------
void Trigger::updateLoop()
{
    // Convert states
    std::vector<double> error_state(last_error_state_.data(), last_error_state_.data() + last_error_state_.size());
    std::vector<double> robot_state(last_robot_state_.data(), last_robot_state_.data() + last_robot_state_.size());
    
    // Evaluate safety and fall status
    is_last_state_safe_ = safety_monitor_->evaluateSafety(edge::config::RuleType::RELATIVE, error_state, obstacle_distance_, goal_distance_);
    is_last_state_fall_ = is_robot_state_initialized_ ? safety_monitor_->detectFall(robot_state) : false;

    if (print_processing_) {
        RCLCPP_INFO(this->get_logger(), "Trigger: is_safe = %s", is_last_state_safe_ ? "true" : "false");
    }

    // Determine terminal mode based on safety status
    edge_msgs::msg::EdgeAction terminal_action;
    edge::utils::ActionMode terminal_mode = is_last_state_safe_ 
                                    ? edge::utils::ActionMode::STUDENT 
                                    : edge::utils::ActionMode::TEACHER;

    // Update switching statistics
    updateStats(terminal_mode);    

    // Decide which action to publish
    if (is_last_state_safe_)
        terminal_action = edge::ros_interfaces::EigenToEdgeActionMsg(last_stu_action_, 
                                                                    edge::utils::ToMsg(terminal_mode));
    else
        terminal_action = edge::ros_interfaces::EigenToEdgeActionMsg(last_tea_action_, 
                                                                    edge::utils::ToMsg(terminal_mode));
    // Publish terminal action
    pub_action_->publish(terminal_action);

    // Publish alert status
    std_msgs::msg::Bool alert_msg;
    alert_msg.data = is_last_state_safe_;
    pub_alert_->publish(alert_msg);

    // Publish fall status
    std_msgs::msg::Bool fall_msg;
    fall_msg.data = is_last_state_fall_;
    pub_fall_->publish(fall_msg);

}

}  // namespace edge::trigger


int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto trigger_node = std::make_shared<edge::trigger::Trigger>(rclcpp::NodeOptions());
    rclcpp::spin(trigger_node);
    rclcpp::shutdown();
    return 0;
}
