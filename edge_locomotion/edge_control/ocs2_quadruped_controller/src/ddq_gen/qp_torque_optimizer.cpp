
#include <torch/torch.h>
#include <torch/script.h>
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <cmath>
#include <stdexcept>
#include <iostream>

// Forward declarations
class MotorCommand; // Assuming this exists
class Robot; // Assuming this exists  
class GaitGenerator; // Assuming this exists

// Utility functions from torch_utils (assuming these exist)
torch::Tensor quat_mul(const torch::Tensor& q1, const torch::Tensor& q2);
torch::Tensor quat_from_euler_xyz(const torch::Tensor& roll, const torch::Tensor& pitch, const torch::Tensor& yaw);
torch::Tensor to_torch(const std::vector<float>& data, const std::string& device = "cuda", torch::ScalarType dtype = torch::kFloat32);
torch::Tensor to_torch(const Eigen::VectorXf& data, const std::string& device = "cuda", torch::ScalarType dtype = torch::kFloat32);
torch::Tensor to_torch(float value, const std::string& device = "cuda", torch::ScalarType dtype = torch::kFloat32);
torch::Tensor quat_rotate(const torch::Tensor& quat, const torch::Tensor& vec);
torch::Tensor quat_to_rot_mat(const torch::Tensor& quat);

/**
 * Solves the centroidal QP to compute desired foot torques.
 */

torch::Tensor cross_quad(const torch::Tensor& v1, const torch::Tensor& v2) {
    // Assumes v1 is nx3, v2 is nx4x3
    torch::Tensor v1_expanded = torch::stack({v1, v1, v1, v1}, 1);
    auto shape = v1_expanded.sizes();
    torch::Tensor v1_reshaped = v1_expanded.reshape({-1, 3});
    torch::Tensor v2_reshaped = v2.reshape({-1, 3});
    return torch::cross(v1_reshaped, v2_reshaped).reshape({shape[0], shape[1], 3});
}

std::tuple<torch::Tensor, torch::Tensor> quaternion_to_axis_angle(const torch::Tensor& q) {
    torch::Tensor angle = 2 * torch::acos(torch::clip(q.select(1, 3), -0.99999, 0.99999)).unsqueeze(1);
    torch::Tensor norm = torch::clip(torch::linalg_norm(q.slice(1, 0, 3), 1), 1e-5, 1).unsqueeze(1);
    torch::Tensor axis = q.slice(1, 0, 3) / norm;
    return std::make_tuple(axis, angle);
}

torch::Tensor compute_orientation_error(const torch::Tensor& desired_orientation_rpy,
                                       const torch::Tensor& base_orientation_quat,
                                       const std::string& device = "cuda") {
    torch::Tensor desired_quat = quat_from_euler_xyz(
        desired_orientation_rpy.select(1, 0), 
        desired_orientation_rpy.select(1, 1),
        torch::zeros_like(desired_orientation_rpy.select(1, 2)));
    
    torch::Tensor base_quat_inv = torch::clone(base_orientation_quat);
    base_quat_inv.select(1, -1) *= -1;
    torch::Tensor error_quat = quat_mul(desired_quat, base_quat_inv);
    
    auto [axis, angle] = quaternion_to_axis_angle(error_quat);
    angle = torch::where(angle > M_PI, angle - 2 * M_PI, angle);
    torch::Tensor error_so3 = axis * angle;
    return quat_rotate(base_orientation_quat, error_so3);
}

torch::Tensor compute_desired_acc(
    const torch::Tensor& base_orientation_rpy,
    const torch::Tensor& base_position,
    const torch::Tensor& base_angular_velocity_body_frame,
    const torch::Tensor& base_velocity_body_frame,
    const torch::Tensor& desired_base_orientation_rpy,
    const torch::Tensor& desired_base_position,
    const torch::Tensor& desired_angular_velocity,
    const torch::Tensor& desired_linear_velocity,
    const torch::Tensor& desired_angular_acceleration,
    const torch::Tensor& desired_linear_acceleration,
    const torch::Tensor& base_position_kp,
    const torch::Tensor& base_position_kd,
    const torch::Tensor& base_orientation_kp,
    const torch::Tensor& base_orientation_kd,
    const std::string& device = "cuda") {
    
    torch::Tensor base_rpy = base_orientation_rpy;
    torch::Tensor base_quat = quat_from_euler_xyz(
        base_rpy.select(1, 0), base_rpy.select(1, 1),
        torch::zeros_like(base_rpy.select(1, 0)));
    
    torch::Tensor base_rot_mat_zero_yaw = quat_to_rot_mat(base_quat);
    torch::Tensor base_rot_mat_zero_yaw_t = base_rot_mat_zero_yaw.transpose(1, 2);

    torch::Tensor lin_pos_error = desired_base_position - base_position;
    lin_pos_error.slice(1, 0, 2).zero_();
    torch::Tensor lin_vel_error = desired_linear_velocity - 
        torch::bmm(base_rot_mat_zero_yaw, base_velocity_body_frame.unsqueeze(2)).squeeze(2);
    torch::Tensor desired_lin_acc_gravity_frame = (base_position_kp * lin_pos_error +
                                                  base_position_kd * lin_vel_error +
                                                  desired_linear_acceleration);

    torch::Tensor ang_pos_error = compute_orientation_error(desired_base_orientation_rpy,
                                                           base_quat,
                                                           device);
    torch::Tensor ang_vel_error = desired_angular_velocity - 
        torch::bmm(base_rot_mat_zero_yaw, base_angular_velocity_body_frame.unsqueeze(2)).squeeze(2);
    torch::Tensor desired_ang_acc_gravity_frame = (base_orientation_kp * ang_pos_error +
                                                  base_orientation_kd * ang_vel_error +
                                                  desired_angular_acceleration);

    torch::Tensor desired_lin_acc_body_frame = 
        torch::bmm(base_rot_mat_zero_yaw_t, desired_lin_acc_gravity_frame.unsqueeze(2)).squeeze(2);
    torch::Tensor desired_ang_acc_body_frame = 
        torch::bmm(base_rot_mat_zero_yaw_t, desired_ang_acc_gravity_frame.unsqueeze(2)).squeeze(2);

    return torch::cat({desired_lin_acc_body_frame, desired_ang_acc_body_frame}, 1);
}

torch::Tensor convert_to_skew_symmetric_batch(const torch::Tensor& foot_positions) {
    /**
     * Converts foot positions (nx4x3) into skew-symmetric ones (nx3x12)
     */
    int64_t n = foot_positions.size(0);
    torch::Tensor x = foot_positions.select(2, 0);  // - 0.023
    torch::Tensor y = foot_positions.select(2, 1);
    torch::Tensor z = foot_positions.select(2, 2);
    torch::Tensor zero = torch::zeros_like(x);
    
    torch::Tensor skew = torch::stack({zero, -z, y, z, zero, -x, -y, x, zero}, 1).reshape({n, 3, 3, 4});
    
    return torch::cat({
        skew.select(3, 0), skew.select(3, 1), skew.select(3, 2), skew.select(3, 3)
    }, 2);
}

torch::Tensor construct_mass_mat(const torch::Tensor& foot_positions,
                                const torch::Tensor& foot_contact_state,
                                const torch::Tensor& inv_mass,
                                const torch::Tensor& inv_inertia,
                                const std::string& device = "cuda",
                                bool mask_noncontact_legs = true) {
    int64_t num_envs = foot_positions.size(0);
    torch::Tensor mass_mat = torch::zeros({num_envs, 6, 12}, torch::device(device));
    
    // Construct mass matrix
    torch::Tensor inv_mass_concat = torch::cat({inv_mass, inv_mass, inv_mass, inv_mass}, 1);
    mass_mat.slice(1, 0, 3) = inv_mass_concat.unsqueeze(0);
    
    torch::Tensor px = convert_to_skew_symmetric_batch(foot_positions);
    mass_mat.slice(1, 3, 6) = torch::bmm(inv_inertia.unsqueeze(0).expand({num_envs, -1, -1}), px);
    
    // Mark out non-contact legs
    if (mask_noncontact_legs) {
        torch::Tensor non_contact_indices = torch::nonzero(torch::logical_not(foot_contact_state));
        if (non_contact_indices.numel() > 0) {
            torch::Tensor env_id = non_contact_indices.select(1, 0);
            torch::Tensor leg_id = non_contact_indices.select(1, 1);
            
            for (int i = 0; i < env_id.size(0); i++) {
                int64_t env_idx = env_id[i].item<int64_t>();
                int64_t leg_idx = leg_id[i].item<int64_t>();
                mass_mat[env_idx].slice(0, 0, 6).select(1, leg_idx * 3) = 0;
                mass_mat[env_idx].slice(0, 0, 6).select(1, leg_idx * 3 + 1) = 0;
                mass_mat[env_idx].slice(0, 0, 6).select(1, leg_idx * 3 + 2) = 0;
            }
        }
    }
    return mass_mat;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> solve_grf(
    const torch::Tensor& mass_mat,
    const torch::Tensor& desired_acc,
    const torch::Tensor& base_rot_mat_t,
    const torch::Tensor& Wq,
    float Wf,
    float foot_friction_coef,
    bool clip_grf,
    const torch::Tensor& foot_contact_state,
    const std::string& device = "cuda") {
    
    int64_t num_envs = mass_mat.size(0);
    torch::Tensor g = torch::zeros({num_envs, 6}, torch::device(device));
    g.select(1, 2) = 9.8;

    g.slice(1, 0, 3) = torch::bmm(base_rot_mat_t, g.slice(1, 0, 3).unsqueeze(2)).squeeze(2);
    torch::Tensor Q = torch::zeros({num_envs, 6, 6}, torch::device(device)) + Wq.unsqueeze(0);
    torch::Tensor Wf_mat = torch::eye(12, torch::device(device)) * Wf;
    torch::Tensor R = torch::zeros({num_envs, 12, 12}, torch::device(device)) + Wf_mat.unsqueeze(0);

    torch::Tensor quad_term = torch::bmm(torch::bmm(mass_mat.transpose(1, 2), Q), mass_mat) + R;
    torch::Tensor linear_term = torch::bmm(torch::bmm(mass_mat.transpose(1, 2), Q),
                                          (g + desired_acc).unsqueeze(2)).squeeze(2);
    
    torch::Tensor grf = torch::bmm(torch::linalg_pinv(quad_term), linear_term.unsqueeze(2)).squeeze(2);

    torch::Tensor base_rot_mat = base_rot_mat_t.transpose(1, 2);
    grf = grf.reshape({-1, 4, 3});
    torch::Tensor grf_world = torch::bmm(base_rot_mat, grf.transpose(1, 2)).transpose(1, 2);
    
    torch::Tensor clipped_motor = torch::logical_or(
        torch::logical_or(grf_world.select(2, 2) < 10, grf_world.select(2, 2) > 130),
        torch::logical_not(foot_contact_state));
    
    if (clip_grf) {
        grf_world.select(2, 2) = torch::clip(grf_world.select(2, 2), 10, 130);
        grf_world.select(2, 2) *= foot_contact_state;
    }

    torch::Tensor friction_force = torch::norm(grf_world.slice(2, 0, 2), 2) + 0.001;
    torch::Tensor max_friction_force = foot_friction_coef * torch::clip(grf_world.select(2, 2), 0);
    torch::Tensor multiplier = torch::where(friction_force < max_friction_force, 
                                           torch::ones_like(friction_force),
                                           max_friction_force / friction_force);

    torch::Tensor clipped_grf = torch::logical_or(multiplier < 1, clipped_motor);

    if (clip_grf) {
        grf_world.slice(2, 0, 2) *= multiplier.unsqueeze(2);
    }
    
    grf = torch::bmm(base_rot_mat_t, grf_world.transpose(1, 2)).transpose(1, 2);
    grf = grf.reshape({-1, 12});

    // Convert to motor torques
    torch::Tensor solved_acc = torch::bmm(mass_mat, grf.unsqueeze(2)).squeeze(2) - g;
    torch::Tensor qp_cost = torch::bmm(
        torch::bmm((solved_acc - desired_acc).unsqueeze(2).transpose(1, 2), Q),
        (solved_acc - desired_acc).unsqueeze(2)).select(2, 0).select(1, 0);
    
    return std::make_tuple(grf, solved_acc, qp_cost, torch::sum(clipped_grf.to(torch::kFloat32), 1));
}

// Note: solve_grf_qpth would require additional QP library integration
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> solve_grf_qpth(
    const torch::Tensor& mass_mat,
    const torch::Tensor& desired_acc,
    const torch::Tensor& base_rot_mat_t,
    const torch::Tensor& Wq,
    float Wf,
    float foot_friction_coef,
    bool clip_grf,
    const torch::Tensor& foot_contact_state,
    const std::string& device = "cuda") {
    
    int64_t num_envs = mass_mat.size(0);
    torch::Tensor g = torch::zeros({num_envs, 6}, torch::device(device));
    g.select(1, 2) = 9.8;

    g.slice(1, 0, 3) = torch::bmm(base_rot_mat_t, g.slice(1, 0, 3).unsqueeze(2)).squeeze(2);
    torch::Tensor Q = torch::zeros({num_envs, 6, 6}, torch::device(device)) + Wq.unsqueeze(0);
    torch::Tensor Wf_mat = torch::eye(12, torch::device(device)) * Wf;
    torch::Tensor R = torch::zeros({num_envs, 12, 12}, torch::device(device)) + Wf_mat.unsqueeze(0);

    torch::Tensor quad_term = torch::bmm(torch::bmm(mass_mat.transpose(1, 2), Q), mass_mat) + R;
    torch::Tensor linear_term = torch::bmm(torch::bmm(mass_mat.transpose(1, 2), Q),
                                          (g + desired_acc).unsqueeze(2)).squeeze(2);

    torch::Tensor G = torch::zeros({mass_mat.size(0), 24, 12}, torch::device(device));
    torch::Tensor h = torch::zeros({mass_mat.size(0), 24}, torch::device(device)) + 1e-3;
    torch::Tensor base_rot_mat = base_rot_mat_t.transpose(1, 2);
    
    for (int leg_id = 0; leg_id < 4; leg_id++) {
        G.select(1, leg_id * 2).select(1, leg_id * 3 + 2) = 1;
        G.select(1, leg_id * 2 + 1).select(1, leg_id * 3 + 2) = -1;

        int row_id = 8 + leg_id * 4;
        int col_id = leg_id * 3;
        
        G.select(1, row_id).select(1, col_id) = 1;
        G.select(1, row_id).select(1, col_id + 2) = -foot_friction_coef;

        G.select(1, row_id + 1).select(1, col_id) = -1;
        G.select(1, row_id + 1).select(1, col_id + 2) = -foot_friction_coef;

        G.select(1, row_id + 2).select(1, col_id + 1) = 1;
        G.select(1, row_id + 2).select(1, col_id + 2) = -foot_friction_coef;

        G.select(1, row_id + 3).select(1, col_id + 1) = -1;
        G.select(1, row_id + 3).select(1, col_id + 2) = -foot_friction_coef;
        
        G.slice(1, leg_id * 2, leg_id * 2 + 2).slice(2, col_id, col_id + 3) = 
            torch::bmm(G.slice(1, leg_id * 2, leg_id * 2 + 2).slice(2, col_id, col_id + 3), base_rot_mat);
        G.slice(1, row_id, row_id + 4).slice(2, col_id, col_id + 3) = 
            torch::bmm(G.slice(1, row_id, row_id + 4).slice(2, col_id, col_id + 3), base_rot_mat);
    }

    torch::Tensor contact_ids = torch::nonzero(foot_contact_state);
    
    if (contact_ids.numel() > 0) {
        for (int i = 0; i < contact_ids.size(0); i++) {
            int64_t env_idx = contact_ids[i][0].item<int64_t>();
            int64_t leg_idx = contact_ids[i][1].item<int64_t>();
            h[env_idx][leg_idx * 2] = 130;
            h[env_idx][leg_idx * 2 + 1] = -10;
        }
    }

    // Note: This would require actual QP solver integration
    // For now, using pinv as fallback
    torch::Tensor grf = torch::bmm(torch::linalg_pinv(quad_term), linear_term.unsqueeze(2)).squeeze(2);
    
    torch::Tensor solved_acc = torch::bmm(mass_mat, grf.unsqueeze(2)).squeeze(2) - g;
    torch::Tensor qp_cost = torch::bmm(
        torch::bmm((solved_acc - desired_acc).unsqueeze(2).transpose(1, 2), Q),
        (solved_acc - desired_acc).unsqueeze(2)).select(2, 0).select(1, 0);

    return std::make_tuple(grf, solved_acc, qp_cost, torch::zeros({mass_mat.size(0)}, torch::device(device)));
}

// Note: solve_grf_quadprog would require quadprog library integration
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> solve_grf_quadprog(
    const torch::Tensor& mass_mat,
    const torch::Tensor& desired_acc,
    const torch::Tensor& base_rot_mat_t,
    const torch::Tensor& Wq,
    const torch::Tensor& Wf,
    float foot_friction_coef,
    bool clip_grf,
    const torch::Tensor& foot_contact_state,
    const std::string& device = "cuda") {
    
    int64_t num_envs = mass_mat.size(0);
    if (num_envs != 1) {
        throw std::runtime_error("Not Supported!");
    }

    // Convert tensors to Eigen matrices for quadprog
    Eigen::MatrixXd mass_mat_eigen = Eigen::Map<Eigen::MatrixXd>(
        mass_mat[0].to(torch::kDouble).to(torch::kCPU).data_ptr<double>(), 6, 12);
    Eigen::VectorXd desired_acc_eigen = Eigen::Map<Eigen::VectorXd>(
        desired_acc[0].to(torch::kDouble).to(torch::kCPU).data_ptr<double>(), 6);
    Eigen::MatrixXd base_rot_mat_t_eigen = Eigen::Map<Eigen::MatrixXd>(
        base_rot_mat_t[0].to(torch::kDouble).to(torch::kCPU).data_ptr<double>(), 3, 3);
    Eigen::MatrixXd Wq_eigen = Eigen::Map<Eigen::MatrixXd>(
        Wq.to(torch::kDouble).to(torch::kCPU).data_ptr<double>(), 6, 6);
    Eigen::VectorXd foot_contact_state_eigen = Eigen::Map<Eigen::VectorXd>(
        foot_contact_state[0].to(torch::kDouble).to(torch::kCPU).data_ptr<double>(), 4);

    Eigen::VectorXd g = Eigen::VectorXd::Zero(6);
    g(2) = 9.8;
    g.head(3) = base_rot_mat_t_eigen * g.head(3);  // Convert to body frame
    
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6, 6) + Wq_eigen;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(12, 12) * Wf.item<double>();
    Eigen::MatrixXd quad_term = mass_mat_eigen.transpose() * Q * mass_mat_eigen + R;
    Eigen::VectorXd linear_term = mass_mat_eigen.transpose() * Q * (g + desired_acc_eigen);

    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(24, 12);
    Eigen::VectorXd h = Eigen::VectorXd::Constant(24, 1e-3);
    Eigen::MatrixXd base_rot_mat = base_rot_mat_t_eigen.transpose();
    
    for (int leg_id = 0; leg_id < 4; leg_id++) {
        G(leg_id * 2, leg_id * 3 + 2) = 1;
        G(leg_id * 2 + 1, leg_id * 3 + 2) = -1;

        int row_id = 8 + leg_id * 4;
        int col_id = leg_id * 3;
        
        G(row_id, col_id) = 1;
        G(row_id, col_id + 2) = -foot_friction_coef;

        G(row_id + 1, col_id) = -1;
        G(row_id + 1, col_id + 2) = -foot_friction_coef;

        G(row_id + 2, col_id + 1) = 1;
        G(row_id + 2, col_id + 2) = -foot_friction_coef;

        G(row_id + 3, col_id + 1) = -1;
        G(row_id + 3, col_id + 2) = -foot_friction_coef;
        
        G.block(leg_id * 2, col_id, 2, 3) = G.block(leg_id * 2, col_id, 2, 3) * base_rot_mat;
        G.block(row_id, col_id, 4, 3) = G.block(row_id, col_id, 4, 3) * base_rot_mat;
    }

    for (int i = 0; i < 4; i++) {
        if (foot_contact_state_eigen(i) > 0.5) {
            h(i * 2) = 130;
            h(i * 2 + 1) = -10;
        }
    }

    // Note: Would need actual quadprog integration here
    // Using pseudo-inverse as fallback
    Eigen::VectorXd grf = quad_term.completeOrthogonalDecomposition().solve(linear_term);
    Eigen::VectorXd solved_acc = mass_mat_eigen * grf - g;
    double qp_cost = (solved_acc - desired_acc_eigen).transpose() * Q * (solved_acc - desired_acc_eigen);
    
    torch::Tensor grf_tensor = torch::from_blob(grf.data(), {1, 12}, torch::kDouble).to(torch::kFloat32).to(torch::device(device));
    torch::Tensor solved_acc_tensor = torch::from_blob(solved_acc.data(), {1, 6}, torch::kDouble).to(torch::kFloat32).to(torch::device(device));
    torch::Tensor qp_cost_tensor = torch::tensor({qp_cost}, torch::device(device));

    return std::make_tuple(grf_tensor, solved_acc_tensor, qp_cost_tensor, torch::zeros({num_envs}, torch::device(device)));
}

class QPTorqueOptimizer {
    /**
     * Centroidal QP controller to optimize for joint torques.
     *
     * Odometry = Zero-yaw frame / Roomba frame.
     */
private:
    std::shared_ptr<Robot> _robot;
    std::shared_ptr<GaitGenerator> _gait_generator;
    std::string _device;
    int64_t _num_envs;
    bool _clip_grf;
    bool _use_full_qp;
    
    torch::Tensor _base_orientation_kp;
    torch::Tensor _base_orientation_kd;
    torch::Tensor _base_position_kp;
    torch::Tensor _base_position_kd;
    torch::Tensor _desired_base_orientation_rpy;
    torch::Tensor _desired_base_position;
    torch::Tensor _desired_linear_velocity;
    torch::Tensor _desired_angular_velocity;
    torch::Tensor _desired_linear_acceleration;
    torch::Tensor _desired_angular_acceleration;
    torch::Tensor _Wq;
    torch::Tensor _Wf;
    float _foot_friction_coef;
    torch::Tensor _inv_mass;
    torch::Tensor _inv_inertia;
    std::pair<double, double> _swing_foot_limit;
    torch::Tensor _last_leg_state;
    torch::Tensor _last_timestamp;
    torch::Tensor _mass_mat;

public:
    QPTorqueOptimizer(std::shared_ptr<Robot> robot,
                     std::shared_ptr<GaitGenerator> gait_generator,
                     const std::vector<double>& base_position_kp = {50., 50., 50},
                     const std::vector<double>& base_position_kd = {1., 1., 1.},
                     const std::vector<double>& base_orientation_kp = {50., 50., 0.},
                     const std::vector<double>& base_orientation_kd = {1., 1., 10.},
                     const std::vector<double>& weight_ddq = {1., 1., 10., 10., 10., 1.},
                     double weight_grf = 1e-4,
                     double body_mass = 14.076,
                     const std::vector<double>& body_inertia = {0.14*1.5, 0.35*1.5, 0.35*1.5},
                     double desired_body_height = 0.26,
                     double foot_friction_coef = 0.7,
                     bool clip_grf = false,
                     const std::pair<double, double>& swing_foot_limit = {-0.4, -0.1},
                     bool use_full_qp = false) 
        : _robot(robot), _gait_generator(gait_generator), _clip_grf(clip_grf), _use_full_qp(use_full_qp) {
        
        // Initialize device and num_envs (assuming these are accessible from robot)
        _device = "cuda"; // robot->device();
        _num_envs = 1; // robot->num_envs();
        
        _base_orientation_kp = to_torch(std::vector<float>(base_orientation_kp.begin(), base_orientation_kp.end()), _device);
        _base_orientation_kp = torch::stack(std::vector<torch::Tensor>(_num_envs, _base_orientation_kp), 0);
        
        _base_orientation_kd = to_torch(std::vector<float>(base_orientation_kd.begin(), base_orientation_kd.end()), _device);
        _base_orientation_kd = torch::stack(std::vector<torch::Tensor>(_num_envs, _base_orientation_kd), 0);
        
        _base_position_kp = to_torch(std::vector<float>(base_position_kp.begin(), base_position_kp.end()), _device);
        _base_position_kp = torch::stack(std::vector<torch::Tensor>(_num_envs, _base_position_kp), 0);
        
        _base_position_kd = to_torch(std::vector<float>(base_position_kd.begin(), base_position_kd.end()), _device);
        _base_position_kd = torch::stack(std::vector<torch::Tensor>(_num_envs, _base_position_kd), 0);
        
        _desired_base_orientation_rpy = torch::zeros({_num_envs, 3}, torch::device(_device));
        _desired_base_position = torch::zeros({_num_envs, 3}, torch::device(_device));
        _desired_base_position.select(1, 2) = desired_body_height;
        
        _desired_linear_velocity = torch::zeros({_num_envs, 3}, torch::device(_device));
        _desired_angular_velocity = torch::zeros({_num_envs, 3}, torch::device(_device));
        _desired_linear_acceleration = torch::zeros({_num_envs, 3}, torch::device(_device));
        _desired_angular_acceleration = torch::zeros({_num_envs, 3}, torch::device(_device));
        
        _Wq = torch::diag(to_torch(std::vector<float>(weight_ddq.begin(), weight_ddq.end()), _device));
        _Wf = to_torch(static_cast<float>(weight_grf), _device);
        _foot_friction_coef = foot_friction_coef;
        _inv_mass = torch::eye(3, torch::device(_device)) / body_mass;
        
        torch::Tensor body_inertia_tensor = torch::diag(to_torch(std::vector<float>(body_inertia.begin(), body_inertia.end()), _device));
        _inv_inertia = torch::linalg_inv(body_inertia_tensor);
        
        _swing_foot_limit = swing_foot_limit;
        
        // Initialize with dummy values - would need actual gait_generator integration
        _last_leg_state = torch::zeros({_num_envs, 4}, torch::device(_device));
        _last_timestamp = torch::zeros({_num_envs}, torch::device(_device));
    }

    void reset() {
        // Reset implementation
    }

    void reset_idx(const std::vector<int>& env_ids) {
        // Reset specific environments
    }

    void update() {
        // compute dt and stance foot odometry position
    }

private:
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
    _solve_joint_torques(const torch::Tensor& foot_contact_state, const torch::Tensor& desired_com_ddq) {
        // Solves centroidal QP to find desired joint torques.
        
        // Note: Would need actual robot interface methods
        torch::Tensor foot_positions_in_base_frame = torch::zeros({_num_envs, 4, 3}, torch::device(_device));
        torch::Tensor base_rot_mat_zero_yaw_t = torch::zeros({_num_envs, 3, 3}, torch::device(_device));
        torch::Tensor all_foot_jacobian = torch::zeros({_num_envs, 12, 12}, torch::device(_device));
        
        _mass_mat = construct_mass_mat(
            foot_positions_in_base_frame,
            foot_contact_state,
            _inv_mass,
            _inv_inertia,
            _device,
            !_use_full_qp);

        // Solve QP
        torch::Tensor grf, solved_acc, qp_cost, num_clips;
        
        if (_use_full_qp) {
            if (_num_envs == 1) {
                std::tie(grf, solved_acc, qp_cost, num_clips) = solve_grf_quadprog(
                    _mass_mat,
                    desired_com_ddq,
                    base_rot_mat_zero_yaw_t,
                    _Wq,
                    _Wf,
                    _foot_friction_coef,
                    _clip_grf,
                    foot_contact_state,
                    _device);
            } else {
                std::tie(grf, solved_acc, qp_cost, num_clips) = solve_grf_qpth(
                    _mass_mat,
                    desired_com_ddq,
                    base_rot_mat_zero_yaw_t,
                    _Wq,
                    _Wf.item<float>(),
                    _foot_friction_coef,
                    _clip_grf,
                    foot_contact_state,
                    _device);
            }
        } else {
            std::tie(grf, solved_acc, qp_cost, num_clips) = solve_grf(
                _mass_mat,
                desired_com_ddq,
                base_rot_mat_zero_yaw_t,
                _Wq,
                _Wf.item<float>(),
                _foot_friction_coef,
                _clip_grf,
                foot_contact_state,
                _device);
        }

        torch::Tensor motor_torques = -torch::bmm(grf.unsqueeze(1), all_foot_jacobian).squeeze(1);
        return std::make_tuple(motor_torques, solved_acc, grf, qp_cost, num_clips);
    }

public:
    torch::Tensor compute_joint_command(const torch::Tensor& foot_contact_state,
                                       const torch::Tensor& desired_base_orientation_rpy,
                                       const torch::Tensor& desired_base_position,
                                       const torch::Tensor& desired_swing_foot_position,
                                       const torch::Tensor& desired_angular_velocity,
                                       const torch::Tensor& desired_linear_velocity,
                                       const torch::Tensor& desired_foot_velocity,
                                       const torch::Tensor& desired_angular_acceleration,
                                       const torch::Tensor& desired_linear_acceleration,
                                       const torch::Tensor& desired_foot_acceleration) {

        // Note: The following commented code would be uncommented in actual implementation
        // desired_base_position = torch::clone(desired_base_position).clip(
        //     robot->base_position() - 0.05, robot->base_position() + 0.05);
        // desired_base_orientation_rpy = torch::clone(
        //     desired_base_orientation_rpy).clip(
        //         robot->base_orientation_rpy() - 0.2,
        //         robot->base_orientation_rpy() + 0.2);

        // Position command
        // Note: More commented code that would need robot interface integration

        // Dummy robot interface values - would need actual implementation
        torch::Tensor base_rot_mat_t = torch::zeros({_num_envs, 3, 3}, torch::device(_device));
        torch::Tensor foot_positions_in_base_frame = torch::zeros({_num_envs, 4, 3}, torch::device(_device));
        torch::Tensor base_rot_mat_zero_yaw_t = torch::zeros({_num_envs, 3, 3}, torch::device(_device));
        torch::Tensor motor_positions = torch::zeros({_num_envs, 12}, torch::device(_device));
        torch::Tensor motor_velocities = torch::zeros({_num_envs, 12}, torch::device(_device));
        torch::Tensor base_orientation_rpy = torch::zeros({_num_envs, 3}, torch::device(_device));
        torch::Tensor base_position = torch::zeros({_num_envs, 3}, torch::device(_device));
        torch::Tensor base_angular_velocity_body_frame = torch::zeros({_num_envs, 3}, torch::device(_device));
        torch::Tensor base_velocity_body_frame = torch::zeros({_num_envs, 3}, torch::device(_device));
        torch::Tensor all_foot_jacobian = torch::zeros({_num_envs, 12, 12}, torch::device(_device));
        torch::Tensor min_positions = torch::zeros({12}, torch::device(_device));
        torch::Tensor max_positions = torch::ones({12}, torch::device(_device));
        torch::Tensor min_torques = torch::zeros({12}, torch::device(_device));
        torch::Tensor max_torques = torch::ones({12}, torch::device(_device));
        
        torch::Tensor desired_swing_foot_position_transformed = torch::bmm(
            base_rot_mat_t,
            desired_swing_foot_position.transpose(1, 2)).transpose(1, 2);
        desired_swing_foot_position_transformed.select(2, 2) = torch::clip(
            desired_swing_foot_position_transformed.select(2, 2),
            _swing_foot_limit.first,
            _swing_foot_limit.second);

        torch::Tensor desired_contact_state = torch::zeros({_num_envs, 4}, torch::device(_device)); // gait_generator->desired_contact_state();
        
        torch::Tensor desired_foot_position = torch::where(
            desired_contact_state.unsqueeze(2).expand({-1, -1, 3}), 
            foot_positions_in_base_frame,
            desired_swing_foot_position_transformed);

        // Note: Would need robot interface for get_motor_angles_from_foot_positions
        torch::Tensor desired_position = torch::zeros({_num_envs, 12}, torch::device(_device));

        desired_position = torch::clip(desired_position,
            min_positions + 1e-3,
            max_positions - 1e-3);
        desired_position = torch::clip(desired_position,
            motor_positions - 0.5,
            motor_positions + 0.5);

        // Velocity Command
        torch::Tensor desired_base_velocity_body_frame = torch::bmm(
            base_rot_mat_zero_yaw_t,
            desired_linear_velocity.unsqueeze(2)).squeeze(2);

        torch::Tensor desired_angular_velocity_body_frame = torch::bmm(
            base_rot_mat_zero_yaw_t,
            desired_angular_velocity.unsqueeze(2)).squeeze(2);

        torch::Tensor desired_foot_velocity_calc = -desired_base_velocity_body_frame.unsqueeze(1) - cross_quad(
            desired_angular_velocity_body_frame,
            foot_positions_in_base_frame);

        torch::Tensor desired_stance_velocity = torch::bmm(
            torch::linalg_pinv(all_foot_jacobian),
            desired_foot_velocity_calc.reshape({_num_envs, 12, 1})).squeeze(2);
        torch::Tensor contact_state_expanded = foot_contact_state.repeat_interleave(3, 1);

        torch::Tensor desired_velocity = torch::where(
            contact_state_expanded, desired_stance_velocity,
            torch::zeros_like(motor_velocities));

        // Torque Command
        torch::Tensor desired_acc_body_frame = compute_desired_acc(
            base_orientation_rpy,
            base_position,
            base_angular_velocity_body_frame,
            base_velocity_body_frame,
            desired_base_orientation_rpy,
            desired_base_position,
            desired_angular_velocity,
            desired_linear_velocity,
            desired_angular_acceleration,
            desired_linear_acceleration,
            _base_position_kp,
            _base_position_kd,
            _base_orientation_kp,
            _base_orientation_kd,
            _device);
        
        desired_acc_body_frame = torch::clip(
            desired_acc_body_frame,
            to_torch(std::vector<float>{-30, -30, -10, -50, -50, -20}, _device),
            to_torch(std::vector<float>{30, 30, 30, 50, 50, 20}, _device));
            
        auto [motor_torques, solved_acc, grf, qp_cost, num_clips] = _solve_joint_torques(
            foot_contact_state, desired_acc_body_frame);
            
        torch::Tensor desired_torque = torch::where(contact_state_expanded, motor_torques,
                                                   torch::zeros_like(motor_torques));
        desired_torque = torch::clip(desired_torque,
                                    min_torques,
                                    max_torques);
        
        return desired_torque;
    }
};

