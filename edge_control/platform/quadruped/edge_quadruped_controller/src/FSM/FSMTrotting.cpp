/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include "edge_msgs/msg/edge_action.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"
#include "edge_quadruped_controller/FSM/FSMTrotting.hpp"
#include <edge_quadruped_controller/utils/math/mathTools.h>
#include <edge_quadruped_controller/control/CtrlComponent.h>
#include <edge_quadruped_controller/estimator/Estimator.hpp>
#include <edge_quadruped_controller/gait/PhaseScheduler.h>

using namespace edge::control::quadruped;

FSMTrotting::FSMTrotting(CtrlInterfaces& ctrl_interfaces,
                             CtrlComponent& ctrl_component,
                             const std::vector<double>& swing_gains,
                             const std::vector<double>& stable_gains,
                             double gait_height) 
                             : FSMState(FSMStateName::TROTTING, "FSM Trotting",
                                ctrl_interfaces),
                                estimator_(ctrl_component.estimator_),
                                robot_model_(ctrl_component.robot_model_),
                                balance_ctrl_(ctrl_component.balance_ctrl_),
                                phase_scheduler_(ctrl_component.phase_scheduler_),
                                ctrl_component_(ctrl_component),
                                gait_generator_(ctrl_component),
                                swing_gains_(swing_gains),
                                stable_gains_(stable_gains),
                                gait_height_(gait_height)
    {
    Kpp = Vec3(70, 70, 70).asDiagonal();
    Kdp = Vec3(20, 20, 20).asDiagonal();
    kp_w_ = 780;
    Kd_w_ = Vec3(70, 70, 70).asDiagonal();
    Kp_swing_ = Vec3(200, 200, 200).asDiagonal();
    Kd_swing_ = Vec3(10, 10, 10).asDiagonal();


    // v_x_limit_ << -1.3, 1.3;        // max forward/backward speed 1.3 m/s (does affect stability)
    // v_y_limit_ << -1.2, 1.2;        // max lateral speed 1.2 m/s (doesn't affect stability much)
    // w_yaw_limit_ << -0.45, 0.45;

    // v_x_limit_ << 0.1, 0.3;        // max forward/backward speed 1.3 m/s (does affect stability)
    // v_y_limit_ << -0.1, 0.1;        // max lateral speed 1.2 m/s (doesn't affect stability much)
    // w_yaw_limit_ << -0.2, 0.2;

    v_x_limit_ << -0.3, 0.3;        // max forward/backward speed 1.3 m/s (does affect stability)
    v_y_limit_ << -0.4, 0.4;        // max lateral speed 1.2 m/s (doesn't affect stability much)
    w_yaw_limit_ << -0.45, 0.45;

    dt_ = 1.0 / ctrl_interfaces_.frequency_;

    // Initialize last tracking error and action
    last_edge_accel_.setZero();
    last_robot_state_.setZero();
    last_tracking_error_tea_.setZero();
    
    curr_edge_accel_.setZero();
    curr_robot_state_.setZero();
    curr_tracking_error_tea_.setZero();

}

void FSMTrotting::enter()
{
    // pcd_ = estimator_->getPosition();
    // pcd_(2) = -estimator_->getFeetPosInBodyFrame()(2, 0);
    pcd_.setZero();  // Reset x,y to zero at the beginning of trotting
    pcd_(2) = 0.3; // Set a default height 0.3m at the beginning
    std::cout << "Enter Trotting State. Pcd initialized to: " << pcd_.transpose() << std::endl;

    v_cmd_body_.setZero();
    yaw_cmd_ = estimator_->getYaw();
    Rd = rotz(yaw_cmd_);
    w_cmd_global_.setZero();

    ctrl_interfaces_.control_inputs_.command = 0;
    gait_generator_.restart();
    
    // Reset step counter for episode tracking
    step_count_ = 0;
}

void FSMTrotting::run(const rclcpp::Time&/*time*/, const rclcpp::Duration&/*period*/, 
                        edge_msgs::msg::EdgeAction& latest_action,
                        rclcpp::Publisher<edge_msgs::msg::Vec12Msg>::SharedPtr tracking_error_tea_pub_, 
                        rclcpp::Publisher<edge_msgs::msg::Vec12Msg>::SharedPtr tracking_error_stu_pub_,
                        rclcpp::Publisher<edge_msgs::msg::Vec12Msg>::SharedPtr robot_state_pub_,
                        rclcpp::Publisher<edge_msgs::msg::Transition>::SharedPtr transition_pub_,
                        double cmd_vx, double cmd_vy, double cmd_wz, double goalX, double goalY,
                        double obstacleAvgDistance, double obstacleMinDistance, bool is_goal_reached, bool is_fall)
{
    // Increment step counter
    ++step_count_;
    
    // Update teacher activation status
    pos_body_ = estimator_->getPosition();
    vel_body_ = estimator_->getVelocity();

    // Robot position in global frame (x, y, yaw)
    robot_pos_(0) = pos_body_(0);
    robot_pos_(1) = pos_body_(1);
    robot_pos_(2) = estimator_->getYaw();

    // Goal position in global frame (x, y)
    nav_goal_(0) = goalX;
    nav_goal_(1) = goalY;

    // std::cout << "Body pos: " << pos_body_.transpose() << std::endl;
    // std::cout << "Body vel: " << vel_body_.transpose() << std::endl;

    B2G_RotMat = estimator_->getRotation();
    G2B_RotMat = B2G_RotMat.transpose();

    getUserCmd(cmd_vx, cmd_vy, cmd_wz);
    calcCmd();

    gait_generator_.setGait(vel_cmd_global_.segment(0, 2), w_cmd_global_(2), gait_height_);
    gait_generator_.generate(pos_feet_global_goal_, vel_feet_global_goal_);

    if (checkStepOrNot())
    {
        phase_scheduler_->status_ = PhaseStatus::WAVE_ALL;
    }
    else
    {
        phase_scheduler_->status_ = PhaseStatus::STANCE_ALL;
    }

    calcTau(tracking_error_tea_pub_, tracking_error_stu_pub_, robot_state_pub_, transition_pub_, latest_action, obstacleAvgDistance, obstacleMinDistance, is_goal_reached, is_fall);
    calcQQd();
    calcGain();
}

void FSMTrotting::exit()
{
    phase_scheduler_->status_ = PhaseStatus::SWING_ALL;
}

FSMStateName FSMTrotting::checkChange()
{
    if (is_reset_)
    {
        is_reset_ = false; // Reset the flag
        return FSMStateName::FIXEDSTAND;
    }

    // if (launch_after_reset_)
    // {
    //     launch_after_reset_ = false; // Reset the flag
    //     return FSMStateName::TROTTING;
    // }

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

void FSMTrotting::getUserCmd(double cmd_vx, double cmd_vy, double cmd_wz)
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

    // Smoothing
    d_yaw_cmd_ = 0.9 * d_yaw_cmd_past_ + (1 - 0.9) * d_yaw_cmd_;
    d_yaw_cmd_past_ = d_yaw_cmd_;
}

void FSMTrotting::calcCmd()
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

void FSMTrotting::calcTau(rclcpp::Publisher<edge_msgs::msg::Vec12Msg>::SharedPtr tracking_error_tea_pub_, 
    rclcpp::Publisher<edge_msgs::msg::Vec12Msg>::SharedPtr tracking_error_stu_pub_, 
    rclcpp::Publisher<edge_msgs::msg::Vec12Msg>::SharedPtr robot_state_pub_,
    rclcpp::Publisher<edge_msgs::msg::Transition>::SharedPtr transition_pub,
    const edge_msgs::msg::EdgeAction& latest_action, double obstacleAvgDistance, double obstacleMinDistance, bool is_goal_reached, bool is_fall)
{
    pos_error_ = pcd_ - pos_body_;
    vel_error_ = vel_cmd_global_ - vel_body_;

    // Vec3 dd_pcd = Kpp * pos_error_ + Kdp * vel_error_;
    // Vec3 d_wbd = kp_w_ * rotMatToExp(Rd * G2B_RotMat) +
        // Kd_w_ * (w_cmd_global_ - estimator_->getGyroGlobal());

    curr_robot_state_ = getRobotState(true);
    curr_tracking_error_tea_ = getTrackingError();
    auto curr_tracking_error_stu_ = getDRLStudentGoalState() - curr_robot_state_;
    // std::cout << "tracking error: ";
    // for (int i = 0; i < 12; i++) {
    //     std::cout << curr_tracking_error_tea_[i] << " ";
    // }
    // std::cout << std::endl;

    // Publish transition topic
    pubTransitionTopic(transition_pub, latest_action, obstacleAvgDistance, obstacleMinDistance, is_goal_reached, is_fall);

    auto track_msg = EigenToVec12Msg(curr_tracking_error_tea_);
    tracking_error_tea_pub_->publish(track_msg);

    auto track_stu_msg = EigenToVec12Msg(curr_tracking_error_stu_);
    tracking_error_stu_pub_->publish(track_stu_msg);

    auto state_msg = EigenToVec12Msg(curr_robot_state_);
    robot_state_pub_->publish(state_msg);

    // std::cout << "rotMatToRPY(Rd * G2B_RotMat)(0): " << rotMatToRPY(Rd * G2B_RotMat)(0) << std::endl;
    // std::cout << "rotMatToExp(Rd * G2B_RotMat)(0): " << rotMatToExp(Rd * G2B_RotMat)(0) << std::endl;

    // std::cout << "pos error: " << pos_error_.transpose() << std::endl;
    // std::cout << "vel error: " << vel_error_.transpose() << std::endl;
    // std::cout << "angle error: " << rotMatToExp(Rd * G2B_RotMat).transpose() << std::endl;
    // std::cout << "angular vel error: " << (w_cmd_global_ - estimator_->getGyroGlobal()).transpose() << std::endl;

    Vec6 accel;
    for (int i = 0; i < 6; i++) {
        accel(i) = latest_action.control_action[i];
    }
    // std::cout << "acc from PHY-Teacher: " << accel.transpose() << std::endl;
    // std::cout << "velocity cmd global: " << vel_cmd_global_.transpose() << std::endl;

    accel(0) = saturation(accel(0), Vec2(-3, 3));
    accel(1) = saturation(accel(1), Vec2(-3, 3));
    accel(2) = saturation(accel(2), Vec2(-5, 5));
    // accel(3) = saturation(accel(3), Vec2(-10, 10));  // Wrong limit, should be (-40, 40)
    // accel(4) = saturation(accel(4), Vec2(-10, 10));  // Wrong limit, should be (-40, 40)
    // accel(5) = saturation(accel(5), Vec2(-20, 20));  // Wrong limit, should be (-10, 10)
    accel(3) = saturation(accel(3), Vec2(-40, 40));
    accel(4) = saturation(accel(4), Vec2(-40, 40));
    accel(5) = saturation(accel(5), Vec2(-20, 20));

    // std::cout << "acc command: " << accel.transpose() << std::endl;

    edge_msgs::msg::EdgeAction latest_action_saturated = EigenToEdgeActionMsg(accel, latest_action.action_mode);

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

    // std::cout << "actual dd_pcd: " << dd_pcd.transpose() << std::endl;
    // std::cout << "actual d_wbd: " << d_wbd.transpose() << std::endl;

    Vec34 force_feet_global =
        -balance_ctrl_->calcGRF(dd_pcd, d_wbd, B2G_RotMat, estimator_->getFeetPosInBodyFrame(), phase_scheduler_->contact_);

    Vec34 pos_feet_global = estimator_->getFeetPos();
    Vec34 vel_feet_global = estimator_->getFeetVel();

    for (int i(0); i < 4; ++i)
    {
        if (phase_scheduler_->contact_(i) == 0)
        {
            force_feet_global.col(i) =
                Kp_swing_ * (pos_feet_global_goal_.col(i) - pos_feet_global.col(i)) +
                Kd_swing_ * (vel_feet_global_goal_.col(i) - vel_feet_global.col(i));
        }
    }

    Vec34 force_feet_body_ = G2B_RotMat * force_feet_global;

    // Send torque command
    for (int i = 0; i < 4; i++)
    {
        Vec3 torque = robot_model_->calcTorque(force_feet_body_.col(i), i);
        for (int j = 0; j < 3; j++)
        {
            ctrl_interfaces_.joint_torque_command_interface_[i * 3 + j].get().set_value(torque(j));
        }
    }

    // update last action and tracking error and robot state
    last_edge_action_ = latest_action_saturated;
    last_tracking_error_tea_ = curr_tracking_error_tea_;  
    last_robot_state_ = curr_robot_state_;
}

void FSMTrotting::calcQQd()
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

void FSMTrotting::calcGain() const
{
    for (int i(0); i < 4; ++i)
    {
        if (phase_scheduler_->contact_(i) == 0)
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

bool FSMTrotting::checkStepOrNot()
{
    // if (fabs(v_cmd_body_(0)) > 0.03 || fabs(v_cmd_body_(1)) > 0.03 ||
    //     fabs(pos_error_(0)) > 0.08 || fabs(pos_error_(1)) > 0.08 ||
    //     fabs(vel_error_(0)) > 0.05 || fabs(vel_error_(1)) > 0.05 ||
    //     fabs(d_yaw_cmd_) > 0.20)
    // {
    //     return true;
    // }
    // return false;
    return true;
}

// Publish transition topic
void FSMTrotting::pubTransitionTopic(rclcpp::Publisher<edge_msgs::msg::Transition>::SharedPtr transition_pub,
                                       const edge_msgs::msg::EdgeAction& latest_action,
                                       double obstacleAvgDistance, double obstacleMinDistance, 
                                       bool is_goal_reached, bool is_fall)
{
    // Skip the first few steps after reset to avoid bad data
    if (step_count_ <= 1) {
        return;
    }

    double time_elapsed = step_count_;  // Use step count as time proxy

    // Check if goal is reached
    auto dist_to_goal = (nav_goal_ - robot_pos_.head<2>()).norm();
    double reach_goal_threshold = 0.4;   // reach_goal threshold

    is_goal_reached = is_goal_reached || (dist_to_goal < reach_goal_threshold);

    // Jerk in action
    auto a1 = EdgeActionMsgToEigen(last_edge_action_);
    auto a2 = EdgeActionMsgToEigen(latest_action);
    Vector6d delta_action = a1 - a2;

    // Prepare reward factors
    edge::config::RewardFactor reward_factors{
        {"last_tracking_error", last_tracking_error_tea_},
        {"current_tracking_error", curr_tracking_error_tea_},
        {"last_robot_state", last_robot_state_},
        {"nav_goal", nav_goal_},
        {"robot_pos", robot_pos_},
        {"robot_state", curr_robot_state_},
        {"is_teacher_activated", latest_action.action_mode == 1 ? true : false},
        {"obstacle_distance_avg", obstacleAvgDistance},
        {"obstacle_distance_min", obstacleMinDistance},
        {"time_elapsed", time_elapsed},
        {"delta_action", delta_action},
        {"is_fall", is_fall},
        {"is_goal_reached", is_goal_reached}
    };

    // Create and populate the Transition message
    edge_msgs::msg::Transition transition;
    transition.obs = getObservation();
    transition.action = getLearnedDRLAction();
    transition.reward_components = reward_composer_.computeTotalReward(reward_factors);
    transition.next_obs = getObservationNext();
    transition.done = is_goal_reached;

    transition_pub->publish(transition);
    // std::cout << "Published to transition topic." << std::endl;
}


Vector12d FSMTrotting::getTrackingError() const
{
    Vector12d tracking_error = Vector12d::Zero();
    
    // Distance to goal
    Vec2 dist_to_goal = nav_goal_ - pos_body_.head<2>();

    // X and Y position error to goal
    // tracking_error[0] = dist_to_goal(0);  // X position error to goal
    // tracking_error[1] = dist_to_goal(1);  // Y position error to goal

    // Height Position error  
    tracking_error[2] = pos_error_(2);

    // Orientation error
    tracking_error.segment<3>(3) = rotMatToExp(Rd * G2B_RotMat);

    // Linear velocity error
    tracking_error.segment<3>(6) = vel_error_;

    // Angular velocity error
    tracking_error.segment<3>(9) = w_cmd_global_ - estimator_->getGyroGlobal();

    return tracking_error;
}


std::vector<double> FSMTrotting::getObservation() const
{
    Vector12d last_base_observation =  getDRLStudentGoalState() - last_robot_state_;
    return EigenToVector(last_base_observation);
}

std::vector<double> FSMTrotting::getObservationNext() const
{
    Vector12d curr_base_observation =  getDRLStudentGoalState() - curr_robot_state_;
    return EigenToVector(curr_base_observation);
}

edge_msgs::msg::EdgeAction FSMTrotting::getLearnedDRLAction() const
{
    /**
     * Calculate DRL action by removing residual effect from teacher action
     * learned_drl_action = terminal_action - residual_gain @ tracking_error
     */
    auto last_residual_action = ctrl_component_.residual_gain_ * last_tracking_error_tea_;
    auto learned_drl_action = EdgeActionMsgToEigen(last_edge_action_) - last_residual_action;

    // std::cout << "last residual action: " << last_residual_action.transpose() << std::endl;
    // std::cout << "learned drl action: " << learned_drl_action.transpose() << std::endl;

    return EigenToEdgeActionMsg(learned_drl_action, last_edge_action_.action_mode);
}

Vector12d FSMTrotting::getRobotState(bool is_global/* = true*/) const
{
    /**
    * Get the robot state in either global frame or body frame
    */
    Vector12d robotState = Vector12d::Zero();

    // Robot Global Position state (x, y, z)
    robotState.segment<3>(0) = pos_body_;

    if (!is_global) {
        // Convert to body frame (px and py are zero in body frame)
        robotState[0] = 0.0;
        robotState[1] = 0.0;
    }

    // Robot Orientation state
    robotState.segment<3>(3) = rotMatToExp(estimator_->getRotation());

    // Linear velocity state
    robotState.segment<3>(6) = vel_body_;

    // Angular velocity state
    robotState.segment<3>(9) = estimator_->getGyroGlobal();

    return robotState;
}

Vector12d FSMTrotting::getDRLStudentGoalState() const
{
    Vector12d robotState = Vector12d::Zero();

    robotState[0] = nav_goal_(0);
    robotState[1] = nav_goal_(1);

    // Robot Position state
    robotState[2] = 0.3; // Set a default height 0.3m

    // Robot Orientation state
    // robotState.segment<3>(3) = Eigen::Vector3d::Zero();

    // Linear velocity state
    // robotState.segment<3>(6) = Eigen::Vector3d(0.8, 0.0, 0.0);
    // robotState[6] = 0.7;

    // Angular velocity state
    // robotState.segment<3>(9) = Eigen::Vector3d::Zero();

    return robotState;
}