// phy_teacher.cpp

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "phy_teacher.hpp"

namespace edge::phy_teacher {
  
PhyTeacher::PhyTeacher(const rclcpp::NodeOptions & options)
: Node("phy_teacher", options) {

    // ------------------------------------------------------
    // 1️⃣ Declare ROS parameters with defaults
    // ------------------------------------------------------
    update_rate_        = this->declare_parameter<int>("update_rate", 200);
    print_process_time_ = this->declare_parameter<bool>("print_process_time", false);

    // PhyTeacher config
    tau_                 = this->declare_parameter<int>("tau", 20);
    chi_                 = this->declare_parameter<double>("chi", 0.25);
    teacher_enable_      = this->declare_parameter<bool>("teacher_enable", true);
    teaching_to_learn_   = this->declare_parameter<bool>("teaching_to_learn", true);
    trigger_type_        = this->declare_parameter<std::string>("trigger_type", "event");

    // Initialize Safety Monitor
    safety_monitor_ = std::make_shared<edge::config::SafetyMonitor>();

    RCLCPP_INFO(this->get_logger(), "Update rate: %d Hz", update_rate_);
    RCLCPP_INFO(this->get_logger(), "Print process time: %s", print_process_time_ ? "true" : "false");

    // Initialize LMI Optimizer
    std::vector<double> default_kp_raw = this->declare_parameter<std::vector<double>>("DEFAULT_KP", std::vector<double>(36, 0.0));
    std::vector<double> default_kd_raw = this->declare_parameter<std::vector<double>>("DEFAULT_KD", std::vector<double>(36, 0.0));

    lmi::Mat6x6 default_kp = edge::ros_interfaces::VectorToEigen<6, 6>(default_kp_raw);
    lmi::Mat6x6 default_kd = edge::ros_interfaces::VectorToEigen<6, 6>(default_kd_raw);
    lmi_optimizer_.set_default_kp(default_kp);
    lmi_optimizer_.set_default_kd(default_kd);

    RCLCPP_INFO(this->get_logger(), "LMI Optimizer initialized.");

    lmi_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(1000 / update_rate_),
      std::bind(&PhyTeacher::lmiCallback, this)
    );

    sub_ = this->create_subscription<edge_msgs::msg::TrackingError>("tracking_error", 10,
      std::bind(&PhyTeacher::errorCallback, this, std::placeholders::_1)
    );

    obstacle_subs_ = this->create_subscription<std_msgs::msg::Float32>("/obstacle_distance", 10,
      std::bind(&PhyTeacher::obstacleDistanceCallback, this, std::placeholders::_1)
    );

    goal_subs_ = this->create_subscription<std_msgs::msg::Float32>("/goal_distance", 10,
      std::bind(&PhyTeacher::goalDistanceCallback, this, std::placeholders::_1)
    );

    pub_ = this->create_publisher<edge_msgs::msg::EdgeAction>("phy_teacher_action", 10);

    RCLCPP_INFO(this->get_logger(), "Phy-Teacher Safety Controller Node initialized.");
  }

PhyTeacher::~PhyTeacher() {}

// Subscription callback to receive tracking error
void PhyTeacher::errorCallback(const edge_msgs::msg::TrackingError::SharedPtr msg) {

    // Update error state
    error_state_ = edge::ros_interfaces::TrackingErrorMsgToEigen(*msg);

    for (int i = 0; i < 10 && i < msg->tracking_error.size(); ++i) {
        lmi_optimizer_.tracking_err[i] = msg->tracking_error[i + 2];
    }

    RCLCPP_INFO(this->get_logger(), "Received tracking error message.");
    got_msg_ = true;
}


void PhyTeacher::obstacleDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg) {
    // Update obstacle distances
    obstacle_distance_ = msg->data;
}

void PhyTeacher::goalDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg) {
    // Update goal distances
    goal_distance_ = msg->data;
}

void PhyTeacher::lmiCallback() {
    // Only proceed if a new message has been received
    if (!got_msg_) return;

    auto err_state_ = edge::ros_interfaces::EigenToVector(error_state_);
    bool is_current_safe = safety_monitor_->evaluateSafety(err_state_, obstacle_distance_, goal_distance_);

    // If teacher is not active
    if (!teacher_active_) {
        // If system is unsafe, activate teacher and set patch center
        if (!is_current_safe) {
        patch_center_ = chi_ * error_state_;
        teacher_active_ = true;

        if (print_process_time_)
            start_time_ = this->now();

        // Update lmi optimizer
        // lmi_optimizer_.update();

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
    edge_msgs::msg::EdgeAction teacher_action_msg;
    if (teacher_active_) {
        // Compute the action according to the center of the safety patch
        edge::ros_interfaces::Vector12d error_from_patch = error_state_ - patch_center_;
        // auto kp_term = lmi_optimizer_.F_kp * error_from_patch.head(6) * 100;
        // auto kd_term = lmi_optimizer_.F_kd * error_from_patch.tail(6) * 100;
        auto kp_term = lmi_optimizer_.default_F_kp * error_from_patch.head(6);
        auto kd_term = lmi_optimizer_.default_F_kd * error_from_patch.tail(6);

        auto result = kp_term + kd_term;
        teacher_action_msg = edge::ros_interfaces::EigenToEdgeActionMsg(result, 
                                                                        edge::ToMsg(edge::ActionMode::TEACHER));
    } else {
        // If teacher is not active, set gains to zero
        teacher_action_msg = edge::ros_interfaces::EigenToEdgeActionMsg(edge::ros_interfaces::Vector6d::Zero(),
                                                                        edge::ToMsg(edge::ActionMode::TEACHER));
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
    rclcpp::spin(std::make_shared<edge::phy_teacher::PhyTeacher>());
    rclcpp::shutdown();
    return 0;
}
