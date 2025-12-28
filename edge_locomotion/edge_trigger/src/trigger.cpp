#include "edge_trigger/trigger.hpp"


using namespace std::chrono_literals;
using std::placeholders::_1;

namespace edge::trigger {

Trigger::Trigger(const rclcpp::NodeOptions & options)
: Node("trigger", options)
{
    RCLCPP_INFO(this->get_logger(), "🔘 The Trigger Component is initialized.");
    
    update_rate_ = this->declare_parameter<double>("update_rate", 150.0);
    print_processing_ = this->declare_parameter<bool>("print_processing", false);

    RCLCPP_INFO(this->get_logger(), "🔘 Trigger update rate is set to %.1f Hz", update_rate_);

    // Initialize Safety Monitor
    safety_monitor_ = std::make_shared<edge::config::SafetyMonitor>();

    // Initialize cached variables
    last_error_state_ = edge::ros_interfaces::TrackingErrorMsgToEigen(edge_msgs::msg::TrackingError());        // initial state
    last_student_action_ = edge::ros_interfaces::EdgeActionMsgToEigen(edge_msgs::msg::EdgeAction());     // initial action
    last_teacher_action_ = edge::ros_interfaces::EdgeActionMsgToEigen(edge_msgs::msg::EdgeAction());     // initial action
    is_last_state_safe_ = true;      // initial alert state

    // Subscribers (state, teacher action, student action, obstacle distance)
    sub_error_ = this->create_subscription<edge_msgs::msg::TrackingError>(
        "tracking_error", 50, std::bind(&Trigger::errorCallback, this, _1));

    sub_teacher_ = this->create_subscription<edge_msgs::msg::EdgeAction>(
        "phy_teacher_action", 50, std::bind(&Trigger::teacherCallback, this, _1));

    sub_student_ = this->create_subscription<edge_msgs::msg::EdgeAction>(
        "drl_student_action", 50, std::bind(&Trigger::studentCallback, this, _1));

    sub_obstacle_distance_ = this->create_subscription<std_msgs::msg::Float32>(
        "/obstacle_min_distance", 50, std::bind(&Trigger::obstacleDistanceCallback, this, _1));

    sub_goal_distance_ = this->create_subscription<std_msgs::msg::Float32>(
        "/goal_distance", 50, std::bind(&Trigger::goalDistanceCallback, this, _1));

    // Publishers (for alert and terminal action)
    pub_alert_ = this->create_publisher<std_msgs::msg::Bool>("is_safe", 50);
    pub_action_ = this->create_publisher<edge_msgs::msg::EdgeAction>("terminal_action", 50);

    auto period = std::chrono::duration<double>(1.0 / update_rate_);
    timer_ = this->create_wall_timer(period, std::bind(&Trigger::updateLoop, this));

}

void Trigger::studentCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg)
{
    last_student_action_ = edge::ros_interfaces::EdgeActionMsgToEigen(*msg);
}

void Trigger::teacherCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg)
{
    last_teacher_action_ = edge::ros_interfaces::EdgeActionMsgToEigen(*msg);
}

void Trigger::errorCallback(const edge_msgs::msg::TrackingError::SharedPtr msg)
{
    last_error_state_ = edge::ros_interfaces::TrackingErrorMsgToEigen(*msg);
    // RCLCPP_INFO(this->get_logger(), "Received new error state: %.3f, %.3f, %.3f",
    //             last_error_state_[0], last_error_state_[1], last_error_state_[2]);
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

// -------------------- Main periodic update --------------------
void Trigger::updateLoop()
{
    std::vector<double> v(last_error_state_.data(), last_error_state_.data() + last_error_state_.size());
    bool is_safe = safety_monitor_->evaluateSafety(edge::config::RuleType::RELATIVE, v, obstacle_distance_, goal_distance_);

    if (print_processing_)
    {
        RCLCPP_INFO(this->get_logger(), "Trigger: is_safe = %s", is_safe ? "true" : "false");
    }

    edge_msgs::msg::EdgeAction terminal_action;
    std_msgs::msg::Bool alert_msg;
    alert_msg.data = is_safe;

    // Decide which action to publish
    if (is_safe)
        terminal_action = edge::ros_interfaces::EigenToEdgeActionMsg(last_student_action_, 
                                                                    edge::ToMsg(edge::ActionMode::STUDENT));
    else
        terminal_action = edge::ros_interfaces::EigenToEdgeActionMsg(last_teacher_action_, 
                                                                    edge::ToMsg(edge::ActionMode::TEACHER));

    pub_action_->publish(terminal_action);
    pub_alert_->publish(alert_msg);

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
