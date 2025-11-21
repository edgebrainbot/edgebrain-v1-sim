#pragma once

#include "control/CtrlComponent.h"
#include "FSM/StateFixedStand.h"
#include "FSM/StateFreeStand.h"
#include "FSM/StateBalanceTest.h"
#include "FSM/StateTrotting.h"

#include "edge_msgs/msg/edge_action.hpp"
#include "edge_msgs/msg/tracking_error.hpp"
#include "edge_msgs/msg/transition.hpp"

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <controller_interface/controller_interface.hpp>
#include <controller_common/FSM/FSMState.h>
#include <controller_common/FSM/StatePassive.h>
#include <controller_common/FSM/StateFixedDown.h>
#include <controller_common/common/enumClass.h>
#include <controller_common/common/mathTypes.h>

namespace edge::locomotion
{
    struct CtrlComponent;
    class WaveGenerator;
    class BalanceCtrl;
    class QuadrupedModel;
    class Estimator;
}

namespace edge::locomotion
{
    struct FSMStateList
    {
        std::shared_ptr<FSMState> invalid;
        std::shared_ptr<StatePassive> passive;
        std::shared_ptr<StateFixedDown> fixedDown;
        std::shared_ptr<StateFixedStand> fixedStand;
        std::shared_ptr<StateFreeStand> freeStand;
        std::shared_ptr<StateTrotting> trotting;
        std::shared_ptr<StateBalanceTest> balanceTest;
    };

    class BodyBalanceController final : public controller_interface::ControllerInterface
    {
    public: 

        controller_interface::InterfaceConfiguration state_interface_configuration() const override;
        controller_interface::InterfaceConfiguration command_interface_configuration() const override;

        controller_interface::return_type update(
            const rclcpp::Time& time, const rclcpp::Duration& period) override;

        controller_interface::CallbackReturn on_init() override;

        controller_interface::CallbackReturn on_configure(
            const rclcpp_lifecycle::State& previous_state) override;
 
        controller_interface::CallbackReturn on_activate(
            const rclcpp_lifecycle::State& previous_state) override;

        controller_interface::CallbackReturn on_deactivate(
            const rclcpp_lifecycle::State& previous_state) override;

        controller_interface::CallbackReturn on_cleanup(
            const rclcpp_lifecycle::State& previous_state) override;

        controller_interface::CallbackReturn on_error(
            const rclcpp_lifecycle::State& previous_state) override;

        controller_interface::CallbackReturn on_shutdown(
            const rclcpp_lifecycle::State& previous_state) override;

        /**
         * Initialize robot model with auto-generated URDF path
         */
        void initializeRobotModel();

        CtrlComponent ctrl_component_;
        CtrlInterfaces ctrl_interfaces_;

    protected:
        std::vector<std::string> joint_names_;
        std::vector<std::string> command_interface_types_;
        std::vector<std::string> state_interface_types_;

        // Robot parameters
        std::string robot_name_;

        std::string imu_name_;
        std::string command_prefix_;
        std::vector<std::string> imu_interface_types_;


        // FR FL RR RL
        std::vector<double> stand_pos_ = {
            0.0, 0.67, -1.3,
            0.0, 0.67, -1.3,
            0.0, 0.67, -1.3,
            0.0, 0.67, -1.3
        };

        std::vector<double> down_pos_ = {
            0.0, 1.3, -2.4,
            0.0, 1.3, -2.4,
            0.0, 1.3, -2.4,
            0.0, 1.3, -2.4
        };

        // stand_gains_[0] = kp, stand_gains_[1] = kd
        std::vector<double> stand_gains_ = {80.0, 3.5};

        // swing_gains_[0] = kp, swing_gains_[1] = kd
        std::vector<double> swing_gains_ = {3.0, 2.0};

        // stable_gains_[0] = kp, stable_gains_[1] = kd
        // std::vector<double> stable_gains_ = {0.8, 0.8};
        std::vector<double> stable_gains_ = {100, 0.8};

        // Gait parameters
        double gait_height_ = 0.08;

        rclcpp::Subscription<control_input_msgs::msg::Inputs>::SharedPtr control_input_sub_;

        std::unordered_map<
            std::string, std::vector<std::reference_wrapper<hardware_interface::LoanedCommandInterface>>*>
        command_interface_map_ = {
            {"effort", &ctrl_interfaces_.joint_torque_command_interface_},
            {"position", &ctrl_interfaces_.joint_position_command_interface_},
            {"velocity", &ctrl_interfaces_.joint_velocity_command_interface_},
            {"kp", &ctrl_interfaces_.joint_kp_command_interface_},
            {"kd", &ctrl_interfaces_.joint_kd_command_interface_}
        };

        std::string topic_ns_;  // Topic namespace
        std::string state_name_;
        FSMStateList state_list_;
        double update_frequency_;
        FSMMode mode_ = FSMMode::NORMAL;
        FSMStateName next_state_name_ = FSMStateName::INVALID;
        std::shared_ptr<FSMState> current_state_;
        std::shared_ptr<FSMState> next_state_;

        std::chrono::time_point<std::chrono::steady_clock> last_update_time_;

        std::shared_ptr<FSMState> getNextState(FSMStateName stateName) const;

        std::unordered_map<
            std::string, std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>>*>
        state_interface_map_ = {
            {"position", &ctrl_interfaces_.joint_position_state_interface_},
            {"effort", &ctrl_interfaces_.joint_effort_state_interface_},
            {"velocity", &ctrl_interfaces_.joint_velocity_state_interface_}
        };

        // For /cmd_vel subscription
        rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_sub_;
        double cmd_vx_ = 0.0;
        double cmd_vy_ = 0.0;
        double cmd_wz_ = 0.0;

        // void cmdVelCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
        // {
        //     cmd_vx_ = msg->twist.linear.x;
        //     cmd_vy_ = msg->twist.linear.y;
        //     cmd_wz_ = msg->twist.angular.z;
        //     std::cout << "*************************************************************" << std::endl;
        //     std::cout << "[StateTrotting] Received /cmd_vel: "
        //             << "vx = " << cmd_vx_ << ", "
        //             << "vy = " << cmd_vy_ << ", "
        //             << "wz = " << cmd_wz_ << std::endl;
        // }

        rclcpp::Publisher<edge_msgs::msg::TrackingError>::SharedPtr tracking_error_pub_;
        rclcpp::Publisher<edge_msgs::msg::Transition>::SharedPtr transition_pub_;
        rclcpp::Subscription<edge_msgs::msg::EdgeAction>::SharedPtr action_sub_;

        edge_msgs::msg::EdgeAction latest_edge_action_;
        // void actionCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg)
        // {
        //     latest_accel_ = *msg;
        //     std::cout << "Received new action." << std::endl;
        //     for (int i = 0; i < 6; i++) {
        //         std::cout << "accel[" << i << "] = " << latest_accel_.action[i] << std::endl;
        //     }
        // }

        // rclcpp::Publisher<phy_teacher_interfaces::msg::TrackingError>::SharedPtr pub_;
        // rclcpp::Subscription<phy_teacher_interfaces::msg::ControlGains>::SharedPtr sub_gain_;

        // void controlGainCallback(const phy_teacher_interfaces::msg::ControlGains::SharedPtr msg) {
        //     std::cout << "Received new control gains." << std::endl;
        //     for (int i = 0; i < 6; i++) {
        //         for (int j = 0; j < 6; j++) {
        //             latest_kp_(i, j) = msg->kp[i*6 + j];
        //             latest_kd_(i, j) = msg->kd[i*6 + j];
        //         }
        //     }
        //     std::cout << "Latest Kp: " << std::endl << latest_kp_ << std::endl;
        //     std::cout << "Latest Kd: " << std::endl << latest_kd_ << std::endl;
        //     gain_update_ = true;
        // }

        Mat6 latest_kp_ = (Mat6() << 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0,
                          0, 0, 14, 0, 0, 0,
                          0, 0, 0, 9, 0, 2,
                          0, 0, 0, 0, 9, -0.5,
                          0, 0, 0, -1.3, 0.39, 15).finished() * 5;
                          
        Mat6 latest_kd_ = (Mat6() << 3.4239, 0, 0, 0, 0, 0,
                          0, 2.755, 0, 0, 0, 0,
                          0, 0, 5.8043, 0, 0, 0,
                          0, 0, 0, 5.4217, 0, 0.0482,
                          0, 0, 0, 0, 5.16, -0.02,
                          0, 0, 0, -0.098, -0.0121, 5.3879).finished() * 5;
        bool gain_update_{false};
    };
} // namespace edge::locomotion::body_balance_controller
