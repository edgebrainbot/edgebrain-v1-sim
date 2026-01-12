#pragma once

#include <cmath>
#include <Eigen/Dense>
#include "edge_config/reward_composer.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"


namespace edge::config {

static const double I10by10[10][10] = {
    {1.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0},
    {0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0},
    {0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0},
    {0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0},
    {0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0,   0.0},
    {0.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0,   0.0},
    {0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0,   0.0},
    {0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.0},
    {0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0},
    {0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   1.0}
};

Eigen::MatrixXd P_ = Eigen::Map<const Eigen::Matrix<double, 10, 10, Eigen::RowMajor>>(&I10by10[0][0]);
Eigen::MatrixXd I_ = Eigen::MatrixXd::Identity(12, 12);

inline void registerAllRewardFunc() {
    auto& R = RewardRegistry::instance();
    
    std::cout << "\n[Registry] ========================================\n";
    std::cout << "[Registry] Registering reward functions...\n";

    // ========== Fall Down ==========
    R.registerReward("fall_down", [](const RewardFactor& rf) {
        int is_fail = std::any_cast<int>(rf.at("is_fail"));
        return -static_cast<double>(is_fail);
    });

    // ========== Linear Velocity Tracking ==========
    R.registerReward("lin_vel_tracking", [](const RewardFactor& rf) {
        const Eigen::VectorXd& err = std::any_cast<const Eigen::VectorXd&>(rf.at("err_prev"));
        double e = err(6)*err(6) + err(7)*err(7) + err(8)*err(8);
        return std::exp(-e / 0.2);
    });

    // ========== Angular Velocity Tracking ==========
    R.registerReward("ang_vel_tracking", [](const RewardFactor& rf) {
        const Eigen::VectorXd& err = std::any_cast<const Eigen::VectorXd&>(rf.at("err_prev"));
        double e = err(9)*err(9) + err(10)*err(10) + err(11)*err(11);
        return std::exp(-e / 0.2);
    });

    // ========== Orientation Tracking ==========
    R.registerReward("orientation_tracking", [](const RewardFactor& rf) {
        const Eigen::VectorXd& err = std::any_cast<const Eigen::VectorXd&>(rf.at("err_prev"));
        return -(err(3)*err(3) + err(4)*err(4) + err(5)*err(5));
    });

    // ========== Height Tracking ==========
    R.registerReward("height_tracking", [](const RewardFactor& rf) {
        const Eigen::VectorXd& err = std::any_cast<const Eigen::VectorXd&>(rf.at("err_prev"));
        return -std::abs(err(2));
    });

    // ========== Jerky Action ==========
    R.registerReward("jerky_action", [](const RewardFactor& rf) {
        const Eigen::VectorXd& delta_action = std::any_cast<const Eigen::VectorXd&>(rf.at("delta_action"));
        return -delta_action.squaredNorm();
    });

    // ========== Forward Speed ==========
    R.registerReward("forward_speed", [](const RewardFactor& rf) {
        double speed = std::any_cast<double>(rf.at("forward_speed"));
        return speed;
    });

    // ========== Body Contact ==========
    R.registerReward("body_contact", [](const RewardFactor& rf) {
        double c = std::any_cast<double>(rf.at("body_contact"));
        return -c;
    });

    // ========== Energy Consumption ==========
    R.registerReward("energy_consumption", [](const RewardFactor& rf) {
        double en = std::any_cast<double>(rf.at("energy"));
        return -en;
    });

    // ========== Upright ==========
    R.registerReward("upright", [](const RewardFactor& rf) {
        double gz = std::any_cast<double>(rf.at("grav_z"));
        return gz;
    });

    // ========== Lyapunov ==========
    R.registerReward("lyapunov", [](const RewardFactor& rf) {
        const Eigen::VectorXd& err_prev = std::any_cast<const Eigen::VectorXd&>(rf.at("err_prev"));
        const Eigen::VectorXd& err_next = std::any_cast<const Eigen::VectorXd&>(rf.at("err_next"));
        
        Eigen::VectorXd sp = err_prev.tail(10);
        Eigen::VectorXd sn = err_next.tail(10);
        
        double v1 = sp.transpose() * P_ * sp;
        double v2 = sn.transpose() * P_ * sn;
        
        // return v1 - v2;
        return -v2;
    });

    // ========== Body State ==========
    R.registerReward("body_state", [](const RewardFactor& rf) {
        const edge::ros_interfaces::Vector12d& state = std::any_cast<const edge::ros_interfaces::Vector12d&>(rf.at("robot_state"));
        
        double res = state.transpose() * I_ * state;
        return -res;
    });

    // ========== Obstacle Average Distance ==========
    R.registerReward("obstacle_avg_distance", [](const RewardFactor& rf) {
        double obs = std::any_cast<double>(rf.at("obstacle_distance_avg"));
        return -1.0 / std::max(obs, 1e-3);
    });

    // ========== Obstacle Minimal Distance ==========
    R.registerReward("obstacle_min_distance", [](const RewardFactor& rf) {
        double obs = std::any_cast<double>(rf.at("obstacle_distance_min"));
        return -1.0 / std::max(obs, 1e-3);
    });

    // ========== Goal Distance ==========
    R.registerReward("goal_distance", [](const RewardFactor& rf) {
        const Eigen::Matrix<double, 2, 1>& goal = std::any_cast<const Eigen::Matrix<double, 2, 1>&>(rf.at("nav_goal"));
        const Eigen::Matrix<double, 3, 1>& robot_pos = std::any_cast<const Eigen::Matrix<double, 3, 1>&>(rf.at("robot_pos"));
        
        double goalX = goal(0);
        double goalY = goal(1);
        double robotX = robot_pos(0);
        double robotY = robot_pos(1);
        double robotYaw = robot_pos(2);
        
        double dx = goalX - robotX;
        double dy = goalY - robotY;
        
        double cos_yaw = std::cos(robotYaw);
        double sin_yaw = std::sin(robotYaw);
        
        double relativeGoalX = dx * cos_yaw + dy * sin_yaw;
        double relativeGoalY = -dx * sin_yaw + dy * cos_yaw;
        
        return -(relativeGoalX * relativeGoalX + relativeGoalY * relativeGoalY);
    });

    // ========== Teacher Activation Penalty ==========
    R.registerReward("teacher_activation_penalty", [](const RewardFactor& rf) {
        bool teacher_penalty = std::any_cast<bool>(rf.at("is_teacher_activated"));
        return teacher_penalty ? -1.0 : 0.0;
    });

    // ========== Student Activation Reward ==========
    R.registerReward("student_activation_reward", [](const RewardFactor& rf) {
        bool teacher = std::any_cast<bool>(rf.at("is_teacher_active"));
        return teacher ? 0.0 : 1.0;
    });
    
    // ========== Time Travel ==========
    R.registerReward("time_travel", [](const RewardFactor& rf) {
        double travelTime = std::any_cast<double>(rf.at("time_elapsed"));
        return -std::log(travelTime + 1.0);
    });

    std::cout << "[Registry] Total registered: " << R.size() << "\n";
    std::cout << "[Registry] ========================================\n\n";
}
   
}
