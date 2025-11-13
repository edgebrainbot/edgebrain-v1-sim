#ifndef STATETROTTING_H
#define STATETROTTING_H
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include "edge_msgs/msg/edge_action.hpp"
#include "edge_msgs/msg/tracking_error.hpp"
#include "edge_msgs/msg/transition.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"
#include "controller_common/FSM/FSMState.h"
#include "edge_config/reward_composer.hpp"

#include <std_msgs/msg/float64_multi_array.hpp>
#include "body_balance_controller/utils/math/mathTypes.h"
#include "body_balance_controller/gait/GaitGenerator.h"

// Forward declarations for ROS2 types
struct CtrlInterfaces;

namespace edge::locomotion
{
    struct CtrlComponent;
    class WaveGenerator;
    class BalanceCtrl;
    class QuadrupedModel;
    class Estimator;
}

using namespace edge::locomotion;

class StateTrotting final : public FSMState {

public:
    explicit StateTrotting(CtrlInterfaces &ctrl_interfaces,
                           CtrlComponent &ctrl_component,
                           const std::vector<double>& swing_gains,
                           const std::vector<double>& stable_gains,
                           double gait_height);

    void enter() override;

    void run(const rclcpp::Time& time, const rclcpp::Duration& period) override {};
    void run(const rclcpp::Time &time, const rclcpp::Duration &period, edge_msgs::msg::EdgeAction &latest_accel, 
             rclcpp::Publisher<edge_msgs::msg::TrackingError>::SharedPtr tracking_error_pub_, 
             rclcpp::Publisher<edge_msgs::msg::Transition>::SharedPtr transition_pub_,
             double cmd_vx, double cmd_vy, double cmd_wz);

    void exit() override;

    FSMStateName checkChange() override;

private:

    // void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);

    void getUserCmd(double, double, double);

    void calcCmd();

    /**
    * Calculate the torque command
    */
    void calcTau(rclcpp::Publisher<edge_msgs::msg::TrackingError>::SharedPtr tracking_error_pub_, 
                 rclcpp::Publisher<edge_msgs::msg::Transition>::SharedPtr transition_pub_,
                 const edge_msgs::msg::EdgeAction& latest_accel_);

    /**
    * Calculate the joint space velocity and acceleration
    */
    void calcQQd();

    /**
    * Calculate the PD gain for the joints
    */
    void calcGain() const;

    /**
     * Check whether the robot should take a step or not
     * @return
     */
    bool checkStepOrNot();

    void pubTransitionTopic(rclcpp::Publisher<edge_msgs::msg::Transition>::SharedPtr transition_pub,
                            edge::ros_interfaces::Vector12d& tracking_error);

    std::shared_ptr<Estimator> &estimator_;
    std::shared_ptr<QuadrupedModel> &robot_model_;
    std::shared_ptr<BalanceCtrl> &balance_ctrl_;
    std::shared_ptr<WaveGenerator> &wave_generator_;

    GaitGenerator gait_generator_;

    // Robot State
    Vec3 pos_body_, vel_body_;
    RotMat B2G_RotMat, G2B_RotMat;

    // Robot command
    Vec3 pcd_;
    Vec3 vel_cmd_global_, v_cmd_body_;
    double dt_;
    double yaw_cmd_{}, d_yaw_cmd_{}, d_yaw_cmd_past_{};
    Vec3 w_cmd_global_;
    Vec34 pos_feet_global_goal_, vel_feet_global_goal_;
    RotMat Rd;

    // Control Parameters
    double gait_height_;
    Vec3 pos_error_, vel_error_;
    Mat3 Kpp, Kdp, Kd_w_;
    double kp_w_;
    Mat3 Kp_swing_, Kd_swing_;
    Vec2 v_x_limit_, v_y_limit_, w_yaw_limit_;

    Vec12 last_tracking_error_;
    Vec12 curr_tracking_error_;

    std::vector<double> last_edge_action_;
    Vec6 curr_edge_action_;

    // Gain parameters
    std::vector<double> swing_gains_;
    std::vector<double> stable_gains_;

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
    //     gain_update_ = true;
    // }

    RewardComposer reward_composer_{""};

    Mat6 latest_kp_;
    Mat6 latest_kd_;
    bool gain_update_{false};
};


#endif //STATETROTTING_H
