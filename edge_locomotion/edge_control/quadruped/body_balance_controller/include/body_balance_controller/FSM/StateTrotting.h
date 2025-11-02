#ifndef STATETROTTING_H
#define STATETROTTING_H

#include "edge_msgs/msg/action.hpp"
#include "edge_msgs/msg/tracking_error.hpp"
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "controller_common/FSM/FSMState.h"
#include "body_balance_controller/control/BalanceCtrl.h"
#include "body_balance_controller/robot_model/QuadrupedModel.hpp"
#include "body_balance_controller/gait/GaitGenerator.h"
#include "body_balance_controller/utils/math/mathTypes.h"

// #include "phy_teacher_interfaces/msg/tracking_error.hpp"
// #include "phy_teacher_interfaces/msg/control_gains.hpp"


class StateTrotting final : public FSMState {
public:
    explicit StateTrotting(CtrlInterfaces &ctrl_interfaces,
                           CtrlComponent &ctrl_component,
                           const std::vector<double>& swing_gains,
                           const std::vector<double>& stable_gains);

    void enter() override;

    void run(const rclcpp::Time& time, const rclcpp::Duration& period) override {};
    void run(const rclcpp::Time &time, const rclcpp::Duration &period, edge_msgs::msg::Action &latest_accel, 
             rclcpp::Publisher<edge_msgs::msg::TrackingError>::SharedPtr tracking_error_pub_, 
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
                 const edge_msgs::msg::Action& latest_accel_);

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

    Mat6 latest_kp_;
    Mat6 latest_kd_;
    bool gain_update_{false};
};


#endif //STATETROTTING_H
