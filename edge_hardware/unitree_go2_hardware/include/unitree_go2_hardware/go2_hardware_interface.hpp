#ifndef GO2_HARDWARE_INTERFACE_HPP
#define GO2_HARDWARE_INTERFACE_HPP

#include "hardware_interface/system_interface.hpp"
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace edge::go2_hardware_interface
{

class Go2HardwareInterface final : public hardware_interface::SystemInterface
{
public:
    Go2HardwareInterface();
    ~Go2HardwareInterface();

    // Override Node LifecycleNodeInterface methods
    hardware_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
    hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
    hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;
    hardware_interface::CallbackReturn on_cleanup(const rclcpp_lifecycle::State& previous_state) override;
    hardware_interface::CallbackReturn on_shutdown(const rclcpp_lifecycle::State& previous_state) override;
    hardware_interface::CallbackReturn on_error(const rclcpp_lifecycle::State& previous_state) override;

    // Override necessary functions from HardwareInterface
    hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo& info) override;
    
    hardware_interface::return_type read(const rclcpp::Time& time, const rclcpp::Duration& period) override;
    hardware_interface::return_type write(const rclcpp::Time& time, const rclcpp::Duration& period) override;
    
    std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
    std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
    hardware_interface::return_type prepare_command_mode_switch(
        const std::vector<std::string>& start_interfaces,
        const std::vector<std::string>& stop_interfaces) override;  

private:
    std::vector<double> joint_kp_command_;
    std::vector<double> joint_kd_command_;
    std::vector<double> joint_torque_command_;
    std::vector<double> joint_position_command_;
    std::vector<double> joint_velocity_command_;

    std::vector<double> joint_state_position_;
    std::vector<double> joint_state_velocity_;
    std::vector<double> joint_state_effort_;

    std::vector<double> imu_states_;
    std::vector<double> foot_force_;
    std::vector<double> high_states_;

    std::unordered_map<std::string, std::vector<std::string> > joint_mode = {
        {"position", {}},
        {"velocity", {}},
        {"effort", {}}
    };


    void initLowCmd();

    void lowStateMessageHandle(const void *messages);

    void highStateMessageHandle(const void *messages);

    void remoteWirelessHandle(const void *messages);

    unitree_go::msg::dds_::LowCmd_ low_cmd_{}; // default init
    unitree_go::msg::dds_::LowState_ low_state_{}; // default init
    unitree_go::msg::dds_::SportModeState_ high_state_{}; // default init

    std::string network_interface_ = "lo";
    int domain_ = 1;
    bool show_foot_force_ = false;

    /*publisher*/
    unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> low_cmd_publisher_;
    
    /*subscriber*/
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> low_state_subscriber_;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> high_state_subscriber_;
};

} // namespace edge::go2_hardware_interface

#endif // GO2_HARDWARE_INTERFACE_HPP