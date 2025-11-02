#include "unitree_go2_hardware/go2_hardware_interface.hpp"

namespace edge::go2_hardware_interface
{

hardware_interface::CallbackReturn Go2HardwareInterface::on_init(const hardware_interface::HardwareInfo& info)
{
    // Call base class on_init to initialize hardware info
    if (hardware_interface::SystemInterface::on_init(info) != 
        hardware_interface::CallbackReturn::SUCCESS)
    {
        return hardware_interface::CallbackReturn::ERROR;
    }
    info_ = info;

    for (const auto& joint : info_.joints)
    {
        for (const auto& interface : joint.state_interfaces)
        {
            joint_interfaces[interface.name].push_back(joint.name);
        }
    }


    if (const auto network_interface_param = info.hardware_parameters.find("network_interface"); network_interface_param
        != info.hardware_parameters.end())
    {
        network_interface_ = network_interface_param->second;
    }
    if (const auto domain_param = info.hardware_parameters.find("domain"); domain_param != info.hardware_parameters.
        end())
    {
        domain_ = std::stoi(domain_param->second);
    }
    if (const auto show_foot_force_param = info.hardware_parameters.find("show_foot_force"); show_foot_force_param !=
        info.hardware_parameters.end())
    {
        show_foot_force_ = show_foot_force_param->second == "true";
    }

    RCLCPP_INFO(rclcpp::get_logger("unitree_hardware"), " network_interface: %s, domain: %d", network_interface_.c_str(), domain_);
    ChannelFactory::Instance()->Init(domain_, network_interface_);

    low_cmd_publisher_ =
        std::make_shared<ChannelPublisher<unitree_go::msg::dds_::LowCmd_>>(
            TOPIC_LOWCMD);
    low_cmd_publisher_->InitChannel();

    lows_tate_subscriber_ =
        std::make_shared<ChannelSubscriber<unitree_go::msg::dds_::LowState_>>(
            TOPIC_LOWSTATE);
    lows_tate_subscriber_->InitChannel(
        [this](auto&& PH1)
        {
            lowStateMessageHandle(std::forward<decltype(PH1)>(PH1));
        },
        1);
    initLowCmd();

    high_state_subscriber_ =
        std::make_shared<ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>>(
            TOPIC_HIGHSTATE);
    high_state_subscriber_->InitChannel(
        [this](auto&& PH1)
        {
            highStateMessageHandle(std::forward<decltype(PH1)>(PH1));
        },
        1);


    return SystemInterface::on_init(info);

    
    size_t num_joints = info.joints.size();

    // Initialize command and state vectors based on the number of joints
    joint_kp_command_.resize(num_joints, 0.0);
    joint_kd_command_.resize(num_joints, 0.0);
    joint_torque_command_.resize(num_joints, 0.0);
    joint_position_command_.resize(num_joints, 0.0);
    joint_velocity_command_.resize(num_joints, 0.0);

    joint_state_position_.resize(num_joints, 0.0);
    joint_state_velocity_.resize(num_joints, 0.0);
    joint_state_effort_.resize(num_joints, 0.0);

    imu_states_.resize(10, 0.0); // IMU: orientation (4) + angular velocity (3) + linear acceleration (3)
    foot_force_.resize(4, 0.0); // foot force for 4 legs
    high_states_.resize(10, 0.0); // high level states

    // 初始化Unitree GO2的LowCmd消息
    initLowCmd();

    return hardware_interface::CallbackReturn::SUCCESS;


} // namespace edge::go2_hardware_interface