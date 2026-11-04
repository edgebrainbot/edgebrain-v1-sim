#include "safety_patch_controller/FSM/StateBalanceTest.h"

#include <safety_patch_controller/SafetyPatchController.h>

#include "safety_patch_controller/common/mathTools.h"
#include "safety_patch_controller/gait/WaveGenerator.h"

StateBalanceTest::StateBalanceTest(CtrlInterfaces &ctrl_interfaces,
                                   CtrlComponent &ctrl_component)
    : FSMState(FSMStateName::BALANCETEST,
               "balance test",
               ctrl_interfaces),
      estimator_(ctrl_component.estimator_),
      robot_model_(ctrl_component.robot_model_),
      balance_ctrl_(ctrl_component.balance_ctrl_),
      wave_generator_(
          ctrl_component.wave_generator_) {
    _xMax = 0.05;
    _xMin = -_xMax;
    _yMax = 0.05;
    _yMin = -_yMax;
    _zMax = 0.04;
    _zMin = -_zMax;
    _yawMax = 20 * M_PI / 180;
    _yawMin = -_yawMax;

    Kp_p_ = Vec3(150, 150, 150).asDiagonal();
    Kd_p_ = Vec3(25, 25, 25).asDiagonal();

    kp_w_ = 200;
    Kd_w_ = Vec3(30, 30, 30).asDiagonal();
}

void StateBalanceTest::enter() {
    pcd_init_ = estimator_->getBasePosition();
    pcd_ = pcd_init_;
    init_rotation_ = estimator_->getRotationMat();
    wave_generator_->status_ = WaveStatus::STANCE_ALL;
}

void StateBalanceTest::run(const rclcpp::Time &/*time*/, const rclcpp::Duration &/*period*/) {
    pcd_(0) = pcd_init_(0) + invNormalize(ctrl_interfaces_.control_inputs_.ly, _xMin, _xMax);
    pcd_(1) = pcd_init_(1) - invNormalize(ctrl_interfaces_.control_inputs_.lx, _yMin, _yMax);
    pcd_(2) = pcd_init_(2) + invNormalize(ctrl_interfaces_.control_inputs_.ry, _zMin, _zMax);

    const float yaw = -invNormalize(ctrl_interfaces_.control_inputs_.rx, _yawMin, _yawMax);
    Rd_ = rotRz(yaw) * init_rotation_;

    for (int i = 0; i < 12; i++) {
        ctrl_interfaces_.joint_kp_command_interface_[i].get().set_value(0.8);
        ctrl_interfaces_.joint_kd_command_interface_[i].get().set_value(0.8);
    }

    calcTorque();
}

void StateBalanceTest::exit() {
    wave_generator_->status_ = WaveStatus::SWING_ALL;
}

FSMStateName StateBalanceTest::checkChange() {
    switch (ctrl_interfaces_.control_inputs_.command) {
        case 1:
            return FSMStateName::FIXEDDOWN;
        case 2:
            return FSMStateName::FIXEDSTAND;
        default:
            return FSMStateName::BALANCETEST;
    }
}

void StateBalanceTest::calcTorque() {
    const auto B2W_Rotation = estimator_->getRotationMat();
    const RotMat W2B_Rotation = B2W_Rotation.transpose();

    const Vec3 pose_body = estimator_->getBasePosition();
    const Vec3 vel_body = estimator_->getBaseVelocity();

    // expected body acceleration
    dd_pcd_ = Kp_p_ * (pcd_ - pose_body) + Kd_p_ * (Vec3(0, 0, 0) - vel_body);

    // expected body angular acceleration
    d_wbd_ = kp_w_ * rotMatToExp(Rd_ * W2B_Rotation) +
             Kd_w_ * (Vec3(0, 0, 0) - estimator_->getGyroscopeWorld());

    // calculate foot force
    const Vec34 pos_feet_in_body_frame = estimator_->getFeetPosInBodyFrame();
    const Vec34 force_feet_global = -balance_ctrl_->calcGroundReactionForces(dd_pcd_, d_wbd_, B2W_Rotation,
                                                         pos_feet_in_body_frame, wave_generator_->contact_);
    const Vec34 force_feet_body = W2B_Rotation * force_feet_global;

    std::vector<KDL::JntArray> current_joints = robot_model_->current_joint_pos_;
    for (int i = 0; i < 4; i++) {
        KDL::JntArray torque = robot_model_->getTorque(force_feet_body.col(i), i);
        for (int j = 0; j < 3; j++) {
            ctrl_interfaces_.joint_torque_command_interface_[i * 3 + j].get().set_value(torque(j));
            ctrl_interfaces_.joint_position_command_interface_[i * 3 + j].get().set_value(
                current_joints[i](j));
        }
    }
}
