#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::Matrix3d;

/// Cubic bezier curve
inline Vector3d cubicBezier(const Vector3d& x0, const Vector3d& x1, double t) {
    double progress = t * t * t + 3 * t * t * (1 - t);
    return x0 + progress * (x1 - x0);
}

/// Generate swing foot trajectory between start, mid and end points
inline Vector3d genSwingFootTrajectory(double phase,
                                       const Vector3d& start_pos,
                                       const Vector3d& mid_pos,
                                       const Vector3d& end_pos) {
    double cutoff = 0.5;
    if (phase < cutoff) {
        return cubicBezier(start_pos, mid_pos, phase / cutoff);
    } else {
        return cubicBezier(mid_pos, end_pos, (phase - cutoff) / (1.0 - cutoff));
    }
}

/// Cross product for angular velocity × hip positions
inline std::vector<Vector3d> crossQuad(const Vector3d& v1,
                                       const std::vector<Vector3d>& v2_list) {
    std::vector<Vector3d> result;
    for (const auto& v2 : v2_list) {
        result.push_back(v1.cross(v2));
    }
    return result;
}

/// Compute desired swing foot positions (world frame, relative to base)
inline std::vector<Vector3d> computeDesiredFootPositions(
        const Matrix3d& base_rot_mat,
        double base_height,
        const std::vector<Vector3d>& hip_positions_in_body_frame,
        const Vector3d& base_velocity_world_frame,
        const Vector3d& base_angular_velocity_body_frame,
        const Vector3d& projected_gravity,
        double desired_base_height,
        double foot_height,
        double foot_landing_clearance,
        double stance_duration,
        const std::vector<double>& normalized_phase,
        const std::vector<Vector3d>& phase_switch_foot_positions)
{
    size_t n_legs = hip_positions_in_body_frame.size();
    std::vector<Vector3d> foot_positions(n_legs);

    // Hip pos in world frame
    std::vector<Vector3d> hip_position(n_legs);
    for (size_t i = 0; i < n_legs; i++) {
        hip_position[i] = base_rot_mat * hip_positions_in_body_frame[i];
    }

    // Mid-air position
    std::vector<Vector3d> mid_position = hip_position;
    for (size_t i = 0; i < n_legs; i++) {
        mid_position  = -base_height + foot_height;
    }

    // Hip velocity
    auto hip_velocity_body = crossQuad(base_angular_velocity_body_frame, hip_positions_in_body_frame);
    std::vector<Vector3d> hip_velocity(n_legs);
    for (size_t i = 0; i < n_legs; i++) {
        hip_velocity[i] = base_velocity_world_frame + base_rot_mat * hip_velocity_body[i];
    }

    // Landing position
    std::vector<Vector3d> land_position(n_legs);
    for (size_t i = 0; i < n_legs; i++) {
        land_position[i] = hip_position[i] + hip_velocity[i] * (stance_duration / 2.0);
        land_position  = std::max(-0.15, std::min(0.15, land_position ));
        land_position  = std::max(-0.08, std::min(0.08, land_position ));
        land_position  = -base_height + foot_landing_clearance;
    }

    // Final foot position using bezier
    for (size_t i = 0; i < n_legs; i++) {
        foot_positions[i] = genSwingFootTrajectory(
            normalized_phase[i],
            phase_switch_foot_positions[i],
            mid_position[i],
            land_position[i]
        );
    }

    return foot_positions;
}

/// Raibert Swing Leg Controller class
class RaibertSwingLegController {
public:
    RaibertSwingLegController(double desired_base_height = 0.26,
                              double foot_landing_clearance = 0.0,
                              double foot_height = 0.15)
        : desired_base_height_(desired_base_height),
          foot_landing_clearance_(foot_landing_clearance),
          foot_height_(foot_height) {}

    void reset(const Matrix3d& base_rot_mat,
               const std::vector<Vector3d>& foot_positions_in_base_frame)
    {
        size_t n_legs = foot_positions_in_base_frame.size();
        phase_switch_foot_positions_.resize(n_legs);
        for (size_t i = 0; i < n_legs; i++) {
            phase_switch_foot_positions_[i] = base_rot_mat * foot_positions_in_base_frame[i];
        }
    }

    std::vector<Vector3d> desiredFootPositions(
        const Matrix3d& base_rot_mat,
        double base_height,
        const std::vector<Vector3d>& hip_positions_in_body_frame,
        const Vector3d& base_velocity_world_frame,
        const Vector3d& base_angular_velocity_body_frame,
        const Vector3d& projected_gravity,
        double stance_duration,
        const std::vector<double>& normalized_phase)
    {
        return computeDesiredFootPositions(
            base_rot_mat,
            base_height,
            hip_positions_in_body_frame,
            base_velocity_world_frame,
            base_angular_velocity_body_frame,
            projected_gravity,
            desired_base_height_,
            foot_height_,
            foot_landing_clearance_,
            stance_duration,
            normalized_phase,
            phase_switch_foot_positions_
        );
    }

private:
    double desired_base_height_;
    double foot_landing_clearance_;
    double foot_height_;
    std::vector<Vector3d> phase_switch_foot_positions_;
};
