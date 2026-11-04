#include "QpTorqueOptimizer.h"


namespace ocs2::legged_robot {
    QpTorqueOptimizer::QpTorqueOptimizer()
        : body_inertia_(3, 3), base_pos_kp_(3), base_pos_kd_(3), base_ang_kp_(3), base_ang_kd_(3), 
          acc_lower_bound_(6), acc_upper_bound_(6), inv_mass_(6, 6), inv_inertia_(3, 3), A_(6, 12), 
          U_(6, 6), V_(12, 12), gravity_(6), grf_(12), desired_acc_(6)
    {
        // Body inertia matrix
        body_inertia_ << 0.1585, 0.0001, -0.0155,
                         0.0001, 0.4686,     0.0,
                        -0.0155,    0.0,  0.5245;

        // Acceleration weighting diagonal
        vector_t weight_diag(6);
        weight_diag << 1., 1., 10., 10., 10., 1.;
        U_ = weight_diag.asDiagonal();

        // Ground reaction force weighting diagonal
        matrix_t I12x12 = matrix_t::Identity(12, 12);
        V_ = I12x12 * weight_grf_;

        // Base position and angular variable gains
        base_pos_kp_ <<  0., 0., 50.;
        base_pos_kd_ << 10., 10., 10.;
        // base_ang_kp_ << 50., 50., 0.;    // roll, pitch, yaw
        base_ang_kp_ << 0., 50., 50.;       // yaw, pitch, roll
        base_ang_kd_ << 10., 10., 10.;

        // Acceleration bounds
        acc_lower_bound_ << -10., -10., -10., -20., -20., -20.;
        acc_upper_bound_ << 10., 10., 10., 20., 20., 20.;

        // Inverse matrices
        matrix_t I3x3 = matrix_t::Identity(3, 3);
        inv_mass_ = I3x3 / body_mass_;
        inv_inertia_ = body_inertia_.inverse();

        // A matrix
        A_.setZero();

        // Gravity vector
        gravity_ << 0., 0., -9.81, 0., 0., 0.;

        // Ground reaction forces vector
        grf_.setZero();

        // Desired acceleration
        desired_acc_.setZero();
    }


    vector_t QpTorqueOptimizer::optimizeTorque(vector_t com_state, vector_t foot_positions, contact_flag_t foot_contact) {

        // Compute the desired torque
        vector_t desired_torque = inv_mass_ * com_state;

        updateDesiredAcceleration(com_state);

        // Apply the torque limits
        desired_torque = desired_torque.cwiseMax(acc_lower_bound_).cwiseMin(acc_upper_bound_);

        solveGroundReactionForces(foot_positions, foot_contact);

        // Return the optimized torque
        return desired_torque;
    }

    vector_t QpTorqueOptimizer::forceToJointTorques(vector_t Jacobian) {
        // Convert ground reaction forces to joint torques using the Jacobian transpose
        vector_t joint_torques = Jacobian.transpose() * grf_;
        std::cout << "Jacobian: " << Jacobian << std::endl;
        std::cout << "Joint torques from GRF: " << joint_torques.transpose() << std::endl;
        return joint_torques;
    }


    void QpTorqueOptimizer::updateDesiredAcceleration(const vector_t& com_state) {
        // com_state = [v, w, p, ypr]

        vector_t position_acc = base_pos_kp_.cwiseProduct(com_state.segment<3>(6))
                               + base_pos_kd_.cwiseProduct(com_state.segment<3>(0));

        vector_t angular_acc = base_ang_kp_.cwiseProduct(com_state.segment<3>(9))
                               + base_ang_kd_.cwiseProduct(com_state.segment<3>(3));

        desired_acc_.segment<3>(0) = position_acc;
        desired_acc_.segment<3>(3) = angular_acc;

        // Clamp the desired acceleration
        desired_acc_.cwiseMax(acc_lower_bound_);
        desired_acc_.cwiseMin(acc_upper_bound_);

        std::cout << "CoM State is: " << com_state.transpose() << std::endl;
        std::cout << "Computed desired acceleration is: " << desired_acc_.transpose() << std::endl;
    }


    void QpTorqueOptimizer::solveGroundReactionForces(vector_t foot_positions, contact_flag_t foot_contact) {

        updateMatrixA(foot_positions, foot_contact);

        matrix_t quad_term = A_.transpose() * U_ * A_ + V_;    // A^T*U*A + V
        matrix_t linear_term = A_.transpose() * U_ * (desired_acc_ - gravity_);    // A^T*U*(ddq - g)

        std::cout << "A_ is: " << A_ << std::endl;
        std::cout << "A_T is: " << A_.transpose() << std::endl;
        std::cout << "U is: " << U_ << std::endl;
        std::cout << "V is: " << V_ << std::endl;
        std::cout << "gravity_ is: " << gravity_ << std::endl;
        std::cout << "desired_acc_ is: " << desired_acc_.transpose() << std::endl;
        std::cout << "quad_term.inverse(): " << quad_term.inverse() << std::endl;

        grf_ = quad_term.inverse() * linear_term;       // f = (A^T*U*A + V)^-1 * A^T*U*(ddq - g)

        std::cout << "quad_term: " << quad_term << std::endl;
        std::cout << "linear_term: " << linear_term << std::endl;
        std::cout << "grf before processing: " << grf_.transpose() << std::endl;

        vector_t friction_force(4);
        vector_t max_friction_force(4);
        vector_t multiplier(4);

        for (int i = 0; i < 4; ++i) {

            std::cout << "foot contact[" << i << "]: " << foot_contact[i] << std::endl;

            int x_idx = i * 3 + 0;
            int y_idx = i * 3 + 1;
            int z_idx = i * 3 + 2;

            // Clip in z direction
            grf_[z_idx] = std::min(std::max(grf_[z_idx], grf_z_lower_bound_), grf_z_upper_bound_);
            grf_[z_idx] *= foot_contact[i];

            // Clip in x-y direction
            double fx = grf_(x_idx);
            double fy = grf_(y_idx);
            friction_force(i) = std::sqrt(fx*fx + fy*fy) + 1e-3;
            max_friction_force(i) = foot_friction_coef_ * std::max(grf_(z_idx), 0.0);

            if (friction_force(i) < max_friction_force(i)) {
                multiplier(i) = 1.0;
            } else {
                multiplier(i) = max_friction_force(i) / friction_force(i);
            }

            grf_(x_idx) *= multiplier(i);
            grf_(y_idx) *= multiplier(i);
        }

        std::cout << "Ground reaction forces are: " << grf_.transpose() << std::endl;
    }

    void QpTorqueOptimizer::updateMatrixA(const vector_t foot_positions, 
                                          const contact_flag_t foot_contact) 
    {
        // Update the A matrix based on foot positions and contact states
        for (int i = 0; i < foot_contact.size(); ++i) {

            // row 3 to 6
            if (foot_contact[i]) {
                A_.block<3, 3>(3, i*3) = inv_inertia_ * skew_symmetric(foot_positions.segment<3>(i*3));
            } else {
                A_.block<3, 3>(3, i*3) << 0, 0, 0,
                                          0, 0, 0,
                                          0, 0, 0;
            }

            // row 0 to 3 
            A_.block<3, 3>(0, i*3) = inv_mass_;

        }           
    }

}
