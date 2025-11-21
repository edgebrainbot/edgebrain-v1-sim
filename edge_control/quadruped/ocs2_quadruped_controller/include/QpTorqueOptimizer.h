#ifndef QPTORQUEOPTIMIZER_H_
#define QPTORQUEOPTIMIZER_H_

#include <memory>
#include <string>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_quadruped_controller/estimator/StateEstimateBase.h>
#include <ocs2_quadruped_controller/interface/LeggedInterface.h>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace ocs2::legged_robot {
    class QpTorqueOptimizer {
        public:
            QpTorqueOptimizer();
            ~QpTorqueOptimizer();

            vector_t optimizeTorque(vector_t com_state, vector_t foot_positions, contact_flag_t foot_contact);
            vector_t forceToJointTorques(vector_t Jacobian);
            void solveGroundReactionForces(vector_t foot_positions, contact_flag_t foot_contact);
            void updateMatrixA(const vector_t foot_positions, const contact_flag_t foot_contact);
            void updateDesiredAcceleration(const vector_t& com_state);

            matrix_t skew_symmetric(const vector_t& vec) {
                matrix_t skew(3, 3);
                skew << 0, -vec(2), vec(1),
                        vec(2), 0, -vec(0),
                        -vec(1), vec(0), 0;
                return skew;
            }

        private:
            double body_mass_ = 15.019;      // Mass of the robot body (kg)
            double leg_length_ = 0.5;         // Length of the robot legs (m)
            double weight_grf_ = 1e-4;       // Weight of the ground reaction force (N)
            double foot_friction_coef_ = 0.7; // Coefficient of friction for the robot feet
            double grf_z_lower_bound_ = 10;   // Lower bound for ground reaction forces in z direction
            double grf_z_upper_bound_ = 130;  // Upper bound for ground reaction forces in z direction

            matrix_t body_inertia_;          // Inertia matrix of the robot body
            matrix_t inv_mass_;              // Inverse mass matrix
            matrix_t inv_inertia_;           // Inverse inertia matrix
            matrix_t A_;                     // A matrix (Mass)
            matrix_t U_;                     // U matrix (Desired accelerations)
            matrix_t V_;                     // V matrix (Ground reaction forces)

            vector_t base_pos_kp_;           // Proportional gain for base position control
            vector_t base_pos_kd_;           // Derivative gain for base linear velocity control
            vector_t base_ang_kp_;           // Proportional gain for base angular position control
            vector_t base_ang_kd_;           // Derivative gain for base angular velocity control

            vector_t desired_acc_;           // Desired acceleration
            vector_t acc_lower_bound_;      // Lower bound for acceleration
            vector_t acc_upper_bound_;      // Upper bound for acceleration

            vector_t gravity_;              // Gravity vector
            vector_t grf_;                  // Ground reaction forces vector

            // contact_flag_t contact_flag_{};
    };

}


#endif // QPTORQUEOPTIMIZER_H_
