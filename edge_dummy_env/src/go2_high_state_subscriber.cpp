#include "edge_dummy_env/go2_high_state_subscriber.hpp"

#include <iostream>
#include <utility>
#include <unistd.h>

namespace edge::dummy_env {

Go2HighStateSubscriber::Go2HighStateSubscriber(std::string network_interface)
    : network_interface_(std::move(network_interface)),
      subscriber_(kTopicHighState) {}

void Go2HighStateSubscriber::Init() {
    unitree::robot::ChannelFactory::Instance()->Init(0, network_interface_);
    std::cout << "Start to subscribe " << kTopicHighState << std::endl;
    subscriber_.InitChannel(
        [this](auto&& msg)
        {
            HandleMessage(std::forward<decltype(msg)>(msg));
        },
        1);
}

bool Go2HighStateSubscriber::HasState() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return has_state_;
}

unitree_go::msg::dds_::SportModeState_ Go2HighStateSubscriber::GetLatestState() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return latest_state_;
}

Go2HighStateSubscriber::Vector12d Go2HighStateSubscriber::GetLatestVec12Eigen() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return ExtractRobotBodyState(latest_state_);
}

Go2HighStateSubscriber::Vec12Msg Go2HighStateSubscriber::GetLatestVec12Msg() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return ExtractRobotBodyStateVec12Msg(latest_state_);
}

void Go2HighStateSubscriber::PrintLatestState() const {
    unitree_go::msg::dds_::SportModeState_ latest;
    bool has_state = false;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        latest = latest_state_;
        has_state = has_state_;
    }

    if (!has_state) {
        std::cout << "[Go2HighStateSubscriber] No state received yet." << std::endl;
        return;
    }

    auto print_state = [](const char* label, const unitree_go::msg::dds_::SportModeState_& state)
    {
        std::cout << label
                  << " position: "
                  << "x=" << state.position()[0] << ", "
                  << "y=" << state.position()[1] << ", "
                  << "z=" << state.position()[2] << " | ";

        std::cout << "rpy: "
                  << "r=" << state.imu_state().rpy()[0] << ", "
                  << "p=" << state.imu_state().rpy()[1] << ", "
                  << "y=" << state.imu_state().rpy()[2] << " | ";

        std::cout << "vel: "
                  << "vx=" << state.velocity()[0] << ", "
                  << "vy=" << state.velocity()[1] << ", "
                  << "vz=" << state.velocity()[2] << " | ";

        std::cout << "gyro: "
                  << "wx=" << state.imu_state().gyroscope()[0] << ", "
                  << "wy=" << state.imu_state().gyroscope()[1] << ", "
                  << "wz=" << state.imu_state().gyroscope()[2] << std::endl;

        std::cout << std::endl;
    };

    print_state("[latest]", latest);
}

void Go2HighStateSubscriber::Run() const {
    while (true) {
        if (HasState()) {
            const auto state = GetLatestState();
            std::cout << "position: "
                      << "x: " << state.position()[0] << ", "
                      << "y: " << state.position()[1] << ", "
                      << "z: " << state.position()[2] << std::endl;

            std::cout << "velocity: "
                      << "vx: " << state.velocity()[0] << ", "
                      << "vy: " << state.velocity()[1] << ", "
                      << "vz: " << state.velocity()[2] << std::endl;

            std::cout << "euler angles: "
                      << "roll: " << state.imu_state().rpy()[0] << ", "
                      << "pitch: " << state.imu_state().rpy()[1] << ", "
                      << "yaw: " << state.imu_state().rpy()[2] << std::endl;

            std::cout << "angular velocity: "
                      << "wx: " << state.imu_state().gyroscope()[0] << ", "
                      << "wy: " << state.imu_state().gyroscope()[1] << ", "
                      << "wz: " << state.imu_state().gyroscope()[2] << std::endl;

            std::cout << std::endl;
        }
        usleep(20000);
    }
}

Go2HighStateSubscriber::Vector12d Go2HighStateSubscriber::ExtractRobotBodyState(
    const unitree_go::msg::dds_::SportModeState_& state) {
    Vector12d vec12 = Vector12d::Zero();
    vec12(0) = state.position()[0];                     // px
    vec12(1) = state.position()[1];                     // py
    vec12(2) = state.position()[2];                     // pz   
    vec12(3) = state.imu_state().rpy()[0];              // roll
    vec12(4) = state.imu_state().rpy()[1];              // pitch
    vec12(5) = state.imu_state().rpy()[2];              // yaw
    vec12(6) = state.velocity()[0];                     // vx
    vec12(7) = state.velocity()[1];                     // vy
    vec12(8) = state.velocity()[2];                     // vz
    vec12(9) = state.imu_state().gyroscope()[0];        // wx
    vec12(10) = state.imu_state().gyroscope()[1];       // wy
    vec12(11) = state.imu_state().gyroscope()[2];       // wz
    return vec12;
}

Go2HighStateSubscriber::Vec12Msg Go2HighStateSubscriber::ExtractRobotBodyStateVec12Msg(
    const unitree_go::msg::dds_::SportModeState_& state) {
    return edge::ros_interfaces::EigenToVec12Msg(ExtractRobotBodyState(state));
}

void Go2HighStateSubscriber::HandleMessage(const void* message) {
    const auto* state_msg = static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);

    std::lock_guard<std::mutex> lock(state_mutex_);
    latest_state_ = *state_msg;
    has_state_ = true;
}

}  // namespace edge::dummy_env
