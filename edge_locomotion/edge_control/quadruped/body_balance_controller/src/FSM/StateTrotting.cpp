
#include "edge_msgs/msg/action.hpp"
#include "edge_msgs/msg/tracking_error.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"
#include "body_balance_controller/FSM/StateTrotting.h"
#include <body_balance_controller/utils/math/mathTools.h>
#include <body_balance_controller/control/CtrlComponent.h>
#include <body_balance_controller/control/Estimator.h>
#include <body_balance_controller/gait/WaveGenerator.h>

using namespace edge::locomotion;

StateTrotting::StateTrotting(CtrlInterfaces& ctrl_interfaces,
                             CtrlComponent& ctrl_component,
                             const std::vector<double>& swing_gains,
                             const std::vector<double>& stable_gains,
                             double gait_height) 
                             : FSMState(FSMStateName::TROTTING, "trotting",
                                ctrl_interfaces),
                                estimator_(ctrl_component.estimator_),
                                robot_model_(ctrl_component.robot_model_),
                                balance_ctrl_(ctrl_component.balance_ctrl_),
                                wave_generator_(ctrl_component.wave_generator_),
                                gait_generator_(ctrl_component),
                                swing_gains_(swing_gains),
                                stable_gains_(stable_gains),
                                gait_height_(gait_height)
    {
    Kpp = Vec3(70, 70, 70).asDiagonal();
    Kdp = Vec3(20, 20, 20).asDiagonal();
    kp_w_ = 780;
    Kd_w_ = Vec3(70, 70, 70).asDiagonal();
    Kp_swing_ = Vec3(400, 400, 400).asDiagonal();
    Kd_swing_ = Vec3(10, 10, 10).asDiagonal();

    v_x_limit_ << -0.4, 0.4;
    v_y_limit_ << -0.3, 0.3;
    w_yaw_limit_ << -0.5, 0.5;
    dt_ = 1.0 / ctrl_interfaces_.frequency_;


    // pub_ = this->create_publisher<phy_teacher_interfaces::msg::TrackingError>("/tracking_err", 10);
    // sub_gain_ = this->create_subscription<phy_teacher_interfaces::msg::ControlGains>(
            // "/control_gain", 10,
            // std::bind(&StateTrotting::controlGainCallback, this, std::placeholders::_1));

    // cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
    //     "/cmd_vel", 10,
    //     std::bind(&StateTrotting::cmdVelCallback, this, std::placeholders::_1));
    // RCLCPP_INFO(this->get_logger(), "[StateTrotting] Listening to /cmd_vel");

}

void StateTrotting::enter()
{
    pcd_ = estimator_->getPosition();
    pcd_(2) = -estimator_->getFeetPos2Body()(2, 0);
    v_cmd_body_.setZero();
    yaw_cmd_ = estimator_->getYaw();
    Rd = rotz(yaw_cmd_);
    w_cmd_global_.setZero();

    ctrl_interfaces_.control_inputs_.command = 0;
    gait_generator_.restart();
}

void StateTrotting::run(const rclcpp::Time&/*time*/, const rclcpp::Duration&/*period*/, edge_msgs::msg::Action& latest_accel_,
                        rclcpp::Publisher<edge_msgs::msg::TrackingError>::SharedPtr tracking_error_pub_, 
                        double cmd_vx, double cmd_vy, double cmd_wz)
{
    pos_body_ = estimator_->getPosition();
    vel_body_ = estimator_->getVelocity();

    std::cout << "Body pos: " << pos_body_.transpose() << std::endl;
    std::cout << "Body vel: " << vel_body_.transpose() << std::endl;

    B2G_RotMat = estimator_->getRotation();
    G2B_RotMat = B2G_RotMat.transpose();

    getUserCmd(cmd_vx, cmd_vy, cmd_wz);
    calcCmd();

    gait_generator_.setGait(vel_cmd_global_.segment(0, 2), w_cmd_global_(2), gait_height_);
    gait_generator_.generate(pos_feet_global_goal_, vel_feet_global_goal_);

    if (checkStepOrNot())
    {
        wave_generator_->status_ = WaveStatus::WAVE_ALL;
    }
    else
    {
        wave_generator_->status_ = WaveStatus::STANCE_ALL;
    }

    calcTau(tracking_error_pub_, latest_accel_);
    calcQQd();
    calcGain();
}

void StateTrotting::exit()
{
    wave_generator_->status_ = WaveStatus::SWING_ALL;
}

FSMStateName StateTrotting::checkChange()
{
    switch (ctrl_interfaces_.control_inputs_.command)
    {
    case 1:
        return FSMStateName::PASSIVE;
    case 2:
        return FSMStateName::FIXEDSTAND;
    default:
        return FSMStateName::TROTTING;
    }
}

void StateTrotting::getUserCmd(double cmd_vx, double cmd_vy, double cmd_wz)
{
    // /* Movement */
    // v_cmd_body_(0) = invNormalize(ctrl_interfaces_.control_inputs_.ly, v_x_limit_(0), v_x_limit_(1));
    // v_cmd_body_(1) = -invNormalize(ctrl_interfaces_.control_inputs_.lx, v_y_limit_(0), v_y_limit_(1));
    // v_cmd_body_(2) = 0;

    // /* Turning */
    // d_yaw_cmd_ = -invNormalize(ctrl_interfaces_.control_inputs_.rx, w_yaw_limit_(0), w_yaw_limit_(1));
    // d_yaw_cmd_ = 0.9 * d_yaw_cmd_past_ + (1 - 0.9) * d_yaw_cmd_;
    // d_yaw_cmd_past_ = d_yaw_cmd_;
    double vx_in = (fabs(cmd_vx) > 1e-4)
                    ? std::clamp(cmd_vx, v_x_limit_(0), v_x_limit_(1))
                    : invNormalize(ctrl_interfaces_.control_inputs_.ly, v_x_limit_(0), v_x_limit_(1));

    double vy_in = (fabs(cmd_vy) > 1e-4)
                    ? std::clamp(cmd_vy, v_y_limit_(0), v_y_limit_(1))
                    : -invNormalize(ctrl_interfaces_.control_inputs_.lx, v_y_limit_(0), v_y_limit_(1));

    double wz_in = (fabs(cmd_wz) > 1e-4)
                    ? std::clamp(cmd_wz, w_yaw_limit_(0), w_yaw_limit_(1))
                    : -invNormalize(ctrl_interfaces_.control_inputs_.rx, w_yaw_limit_(0), w_yaw_limit_(1));
    v_cmd_body_ << vx_in, vy_in, 0.0;

    d_yaw_cmd_ = wz_in;
    // 滤波
    d_yaw_cmd_ = 0.9 * d_yaw_cmd_past_ + (1 - 0.9) * d_yaw_cmd_;
    d_yaw_cmd_past_ = d_yaw_cmd_;
}

void StateTrotting::calcCmd()
{
    /* Movement */
    vel_cmd_global_ = B2G_RotMat * v_cmd_body_;

    vel_cmd_global_(0) =
        saturation(vel_cmd_global_(0), Vec2(vel_body_(0) - 0.2, vel_body_(0) + 0.2));
    vel_cmd_global_(1) =
        saturation(vel_cmd_global_(1), Vec2(vel_body_(1) - 0.2, vel_body_(1) + 0.2));

    pcd_(0) = saturation(pcd_(0) + vel_cmd_global_(0) * dt_,
                         Vec2(pos_body_(0) - 0.05, pos_body_(0) + 0.05));
    pcd_(1) = saturation(pcd_(1) + vel_cmd_global_(1) * dt_,
                         Vec2(pos_body_(1) - 0.05, pos_body_(1) + 0.05));

    vel_cmd_global_(2) = 0;

    /* Turning */
    yaw_cmd_ = yaw_cmd_ + d_yaw_cmd_ * dt_;
    Rd = rotz(yaw_cmd_);
    w_cmd_global_(2) = d_yaw_cmd_;
}

void StateTrotting::calcTau(rclcpp::Publisher<edge_msgs::msg::TrackingError>::SharedPtr tracking_error_pub_, const edge_msgs::msg::Action& latest_accel_)
{
    pos_error_ = pcd_ - pos_body_;
    vel_error_ = vel_cmd_global_ - vel_body_;

    // Vec3 dd_pcd = Kpp * pos_error_ + Kdp * vel_error_;
    // Vec3 d_wbd = kp_w_ * rotMatToExp(Rd * G2B_RotMat) +
        // Kd_w_ * (w_cmd_global_ - estimator_->getGyroGlobal());

    edge::ros_interfaces::Vector12d tracking_error;
    tracking_error[0] = 0.0;
    tracking_error[1] = 0.0;
    tracking_error[2] = pos_error_(2);
    tracking_error[3] = rotMatToExp(Rd * G2B_RotMat)(0);
    tracking_error[4] = rotMatToExp(Rd * G2B_RotMat)(1);
    tracking_error[5] = rotMatToExp(Rd * G2B_RotMat)(2);
    tracking_error[6] = vel_error_(0);
    tracking_error[7] = vel_error_(1);
    tracking_error[8] = vel_error_(2);
    tracking_error[9] = (w_cmd_global_ - estimator_->getGyroGlobal())(0);
    tracking_error[10] = (w_cmd_global_ - estimator_->getGyroGlobal())(1);
    tracking_error[11] = (w_cmd_global_ - estimator_->getGyroGlobal())(2);

    std::cout << "tracking error: ";
    for (int i = 0; i < 12; i++) {
        std::cout << tracking_error[i] << " ";
    }
    std::cout << std::endl;

    auto track_msg = edge::ros_interfaces::EigenToTrackingErrorMsg(tracking_error);
    tracking_error_pub_->publish(track_msg);

    std::cout << "rotMatToRPY(Rd * G2B_RotMat)(0): " << rotMatToRPY(Rd * G2B_RotMat)(0) << std::endl;
    std::cout << "rotMatToExp(Rd * G2B_RotMat)(0): " << rotMatToExp(Rd * G2B_RotMat)(0) << std::endl;

    // std::cout << "pos error: " << pos_error_.transpose() << std::endl;
    // std::cout << "vel error: " << vel_error_.transpose() << std::endl;
    // std::cout << "angle error: " << rotMatToExp(Rd * G2B_RotMat).transpose() << std::endl;
    // std::cout << "angular vel error: " << (w_cmd_global_ - estimator_->getGyroGlobal()).transpose() << std::endl;

    Vec6 accel;
    for (int i = 0; i < 6; i++) {
        accel(i) = latest_accel_.action[i];
    }
    std::cout << "acc from PHY-Teacher: " << accel.transpose() << std::endl;
    std::cout << "velocity cmd global: " << vel_cmd_global_.transpose() << std::endl;

    // std::cout << "kp: " << kp << std::endl;
    // std::cout << "kd: " << kd << std::endl;

    accel(0) = saturation(accel(0), Vec2(-3, 3));
    accel(1) = saturation(accel(1), Vec2(-3, 3));
    accel(2) = saturation(accel(2), Vec2(-5, 5));
    // accel(3) = saturation(accel(3), Vec2(-10, 10));  // Wrong limit, should be (-40, 40)
    // accel(4) = saturation(accel(4), Vec2(-10, 10));  // Wrong limit, should be (-40, 40)
    // accel(5) = saturation(accel(5), Vec2(-20, 20));  // Wrong limit, should be (-10, 10)
    accel(3) = saturation(accel(3), Vec2(-40, 40));
    accel(4) = saturation(accel(4), Vec2(-40, 40));
    accel(5) = saturation(accel(5), Vec2(-10, 10));

    std::cout << "acc command: " << accel.transpose() << std::endl;

    Vec3 dd_pcd = accel.segment(0, 3);
    Vec3 d_wbd = accel.segment(3, 3);

    // Vec6 all_pos_error;
    // Vec6 all_vel_error;
    // all_pos_error << pos_error_, rotMatToExp(Rd * G2B_RotMat);
    // all_vel_error << vel_error_, (w_cmd_global_ - estimator_->getGyroGlobal());
    // std::cout << "all pos error: " << all_pos_error.transpose() << std::endl;
    // std::cout << "all vel error: " << all_vel_error.transpose() << std::endl;
    // Eigen::Matrix<double , 6, 6> kp;
    // Eigen::Matrix<double , 6, 6> kd;
    // kp << 1, 0, 0, 0, 0, 0,
    //       0, 1, 0, 0, 0, 0,
    //       0, 0, 70, 0, 0, 0,
    //       0, 0, 0, 780, 0, 0,
    //       0, 0, 0, 0, 780, 0,
    //       0, 0, 0, 0, 0, 100;

    // kd << 20, 0, 0, 0, 0, 0,
    //       0, 20, 0, 0, 0, 0,
    //       0, 0, 20, 0, 0, 0,
    //       0, 0, 0, 70, 0, 0,
    //       0, 0, 0, 0, 70, 0,
    //       0, 0, 0, 0, 0, 70;
    // auto acc = kp * all_pos_error + kd * all_vel_error;
    // Vec3 dd_pcd = acc.segment(0, 3);
    // Vec3 d_wbd = acc.segment(3, 3);

    // Vec3 dd_pcd = Kpp * pos_error_ + Kdp * vel_error_;
    // Vec3 d_wbd = kp_w_ * rotMatToExp(Rd * G2B_RotMat) +
    //     Kd_w_ * (w_cmd_global_ - estimator_->getGyroGlobal());

    // dd_pcd(0) = saturation(dd_pcd(0), Vec2(-3, 3));
    // dd_pcd(1) = saturation(dd_pcd(1), Vec2(-3, 3));
    // dd_pcd(2) = saturation(dd_pcd(2), Vec2(-5, 5));

    // d_wbd(0) = saturation(d_wbd(0), Vec2(-40, 40));
    // d_wbd(1) = saturation(d_wbd(1), Vec2(-40, 40));
    // d_wbd(2) = saturation(d_wbd(2), Vec2(-10, 10));

    std::cout << "actual dd_pcd: " << dd_pcd.transpose() << std::endl;
    std::cout << "actual d_wbd: " << d_wbd.transpose() << std::endl;

    Vec34 force_feet_global =
        -balance_ctrl_->calcGRF(dd_pcd, d_wbd, B2G_RotMat, estimator_->getFeetPos2Body(), wave_generator_->contact_);


    Vec34 pos_feet_global = estimator_->getFeetPos();
    Vec34 vel_feet_global = estimator_->getFeetVel();

    for (int i(0); i < 4; ++i)
    {
        if (wave_generator_->contact_(i) == 0)
        {
            force_feet_global.col(i) =
                Kp_swing_ * (pos_feet_global_goal_.col(i) - pos_feet_global.col(i)) +
                Kd_swing_ * (vel_feet_global_goal_.col(i) - vel_feet_global.col(i));
        }
    }

    Vec34 force_feet_body_ = G2B_RotMat * force_feet_global;

    for (int i = 0; i < 4; i++)
    {
        Vec3 torque = robot_model_->calcTorque(force_feet_body_.col(i), i);
        for (int j = 0; j < 3; j++)
        {
            ctrl_interfaces_.joint_torque_command_interface_[i * 3 + j].get().set_value(torque(j));
        }
    }
}

void StateTrotting::calcQQd()
{
    const Vec34 pos_feet_body = robot_model_->getFeetPosInBodyFrame();

    Vec34 pos_feet_target, vel_feet_target;
    for (int i(0); i < 4; ++i)
    {
        pos_feet_target.col(i) = G2B_RotMat * (pos_feet_global_goal_.col(i) - pos_body_);
        vel_feet_target.col(i) = G2B_RotMat * (vel_feet_global_goal_.col(i) - vel_body_);
    }

    Vec12 q_goal = robot_model_->getJointPos(pos_feet_target, FrameType::BODY);
    Vec12 qd_goal = robot_model_->getJointVel(pos_feet_body, vel_feet_target, FrameType::BODY);

    for (int i = 0; i < 12; i++)
    {
        ctrl_interfaces_.joint_position_command_interface_[i].get().set_value(q_goal(i));
        ctrl_interfaces_.joint_velocity_command_interface_[i].get().set_value(qd_goal(i));
    }
}

void StateTrotting::calcGain() const
{
    for (int i(0); i < 4; ++i)
    {
        if (wave_generator_->contact_(i) == 0)
        {
            // swing gain
            for (int j = 0; j < 3; j++)
            {
                ctrl_interfaces_.joint_kp_command_interface_[i * 3 + j].get().set_value(swing_gains_[0]);
                ctrl_interfaces_.joint_kd_command_interface_[i * 3 + j].get().set_value(swing_gains_[1]);
            }
        }
        else
        {
            // stable gain
            for (int j = 0; j < 3; j++)
            {
                ctrl_interfaces_.joint_kp_command_interface_[i * 3 + j].get().set_value(stable_gains_[0]);
                ctrl_interfaces_.joint_kd_command_interface_[i * 3 + j].get().set_value(stable_gains_[1]);
            }
        }
    }
}

bool StateTrotting::checkStepOrNot()
{
    if (fabs(v_cmd_body_(0)) > 0.03 || fabs(v_cmd_body_(1)) > 0.03 ||
        fabs(pos_error_(0)) > 0.08 || fabs(pos_error_(1)) > 0.08 ||
        fabs(vel_error_(0)) > 0.05 || fabs(vel_error_(1)) > 0.05 ||
        fabs(d_yaw_cmd_) > 0.20)
    {
        return true;
    }
    return false;
}
