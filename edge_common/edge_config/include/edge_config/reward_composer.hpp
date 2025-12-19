#pragma once
#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

// static const double MATRIX_P[10][10] = {
//     {140.6434, 0.0,     0.0,     0.0,     0.0,    0.0,    5.3276, 0.0,     0.0,     0.0},
//     {0.0,     134.7596, 0.0,     0.0,     0.0,    0.0,    0.0,    6.6219, 0.0,     0.0},
//     {0.0,     0.0,     134.7596, 0.0,     0.0,    0.0,    0.0,    0.0,    6.622,  0.0},
//     {0.0,     0.0,     0.0,     49.641,  0.0,    0.0,    0.0,    0.0,    0.0,     6.8662},
//     {0.0,     0.0,     0.0,      0.0,    11.1111,0.0,    0.0,    0.0,    0.0,     0.0},
//     {0.0,     0.0,     0.0,      0.0,     0.0,   3.3058, 0.0,    0.0,    0.0,     0.0},
//     {5.3276,  0.0,     0.0,      0.0,     0.0,   0.0,    3.6008, 0.0,    0.0,     0.0},
//     {0.0,     6.6219,  0.0,      0.0,     0.0,   0.0,    0.0,    3.6394, 0.0,     0.0},
//     {0.0,     0.0,     6.622,    0.0,     0.0,   0.0,    0.0,    0.0,    3.6394, 0.0},
//     {0.0,     0.0,     0.0,      0.0,     0.0,   6.8662, 0.0,    0.0,    0.0,     4.3232}
// };

static const double MATRIX_P[10][10] = {
    {1.0,     0.0,     0.0,     0.0,     0.0,    0.0,    0.0,    0.0,    0.0,    0.0},
    {0.0,     1.0,     0.0,     0.0,     0.0,    0.0,    0.0,    0.0,    0.0,    0.0},
    {0.0,     0.0,     1.0,     0.0,     0.0,    0.0,    0.0,    0.0,    0.0,    0.0},
    {0.0,     0.0,     0.0,     1.0,     0.0,    0.0,    0.0,    0.0,    0.0,    0.0},
    {0.0,     0.0,     0.0,     0.0,     1.0,    0.0,    0.0,    0.0,    0.0,    0.0},
    {0.0,     0.0,     0.0,     0.0,     0.0,    1.0,    0.0,    0.0,    0.0,    0.0},
    {0.0,     0.0,     0.0,     0.0,     0.0,    0.0,    1.0,    0.0,    0.0,    0.0},
    {0.0,     0.0,     0.0,     0.0,     0.0,    0.0,    0.0,    1.0,    0.0,    0.0},
    {0.0,     0.0,     0.0,     0.0,     0.0,    0.0,    0.0,    0.0,    1.0,    0.0},
    {0.0,     0.0,     0.0,     0.0,     0.0,    0.0,    0.0,    0.0,    0.0,    1.0}
};

class RewardComposer {
public:
    RewardComposer(const std::string& yaml_path);

    // Compute total reward (scalar only)
    double computeTotalReward(
        const Eigen::VectorXd& err_prev,     // 12D tracking error
        const Eigen::VectorXd& err_next,     // 12D tracking error next
        const Eigen::VectorXd& goal,         // 3D goal position
        const Eigen::VectorXd& vehicle_pos    // 3D vehicle position
        // int is_fail,                          // 0 or 1
        // const Eigen::VectorXd& delta_action, // action_t - action_(t-1)
        // double forward_speed,
        // double body_contact,
        // double energy,
        // double grav_z                         // projected gravity z
    );

private:
    std::unordered_map<std::string, double> scales_;
    Eigen::MatrixXd P_;   // Lyapunov

    // Reward funcs return double now
    double fall_down(int is_fail);
    double lin_vel_tracking(const Eigen::VectorXd& err);
    double ang_vel_tracking(const Eigen::VectorXd& err);
    double orientation_tracking(const Eigen::VectorXd& err);
    double height_tracking(const Eigen::VectorXd& err);
    double jerky_action(const Eigen::VectorXd& delta_action);
    double forward_speed(double speed);
    double body_contact(double c);
    double energy_consumption(double en);
    double upright(double gz);

    double lyapunov(
        const Eigen::VectorXd& err_prev,
        const Eigen::VectorXd& err_next
    );

    double goal_distance(
        const Eigen::VectorXd& goal, 
        const Eigen::VectorXd& vehicle_pos
    );
};
