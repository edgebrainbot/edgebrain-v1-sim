#include "unitree_go2_hardware/crc32.hpp"
#include "unitree_go2_hardware/go2_hardware_interface.hpp"

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_HIGHSTATE "rt/sportmodestate"

namespace edge::go2_hardware_interface
{

/***************************************************************************************/
/*******                                                                         *******/
/*******                             on_init                                     *******/
/*******                                                                         *******/
/***************************************************************************************/
hardware_interface::CallbackReturn Go2HardwareInterface::on_init(
    const hardware_interface::HardwareInfo& info)
{
    // Call base class on_init to initialize hardware info
    if (hardware_interface::SystemInterface::on_init(info) != 
        hardware_interface::CallbackReturn::SUCCESS)
    {
        return hardware_interface::CallbackReturn::ERROR;
    }
    info_ = info;

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

    imu_states_.resize(14, 0.0);    // IMU: orientation (4) + angular velocity (3) + linear acceleration (3)
                                    //      + rpy (3) + temperature (1)
    foot_force_.resize(4, 0.0);     // foot force for 4 legs
    odom_states_.resize(6, 0.0);    // odometry states

    // Populate joint_interface_map
    for (const auto& joint : info_.joints)
    {
        for (const auto& interface : joint.state_interfaces)
        {
            joint_interface_map_[interface.name].push_back(joint.name);
        }
    }

    // Network interface parameters
    if (const auto network_interface_param = info.hardware_parameters.find("network_interface"); 
        network_interface_param != info.hardware_parameters.end())
    {
        network_interface_ = network_interface_param->second;
    }

    // Domain parameter
    if (const auto domain_param = info.hardware_parameters.find("domain"); 
        domain_param != info.hardware_parameters.end())
    {
        domain_ = std::stoi(domain_param->second);
    }

    // Show foot force parameter
    if (const auto show_foot_force_param = info.hardware_parameters.find("show_foot_force"); 
        show_foot_force_param != info.hardware_parameters.end())
    {
        show_foot_force_ = show_foot_force_param->second == "true";
    }

    RCLCPP_INFO(rclcpp::get_logger(logger_name_), " network_interface: %s, domain: %d", network_interface_.c_str(), domain_);
    
    // Initialize communication channels
    unitree::robot::ChannelFactory::Instance()->Init(domain_, network_interface_);

    low_cmd_publisher_ = std::make_shared<unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>>(TOPIC_LOWCMD);
    low_cmd_publisher_->InitChannel();

    low_state_subscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>>(TOPIC_LOWSTATE);
    low_state_subscriber_->InitChannel(
        [this](auto&& msg)
        {
            lowStateMessageHandler(std::forward<decltype(msg)>(msg));
        }, 1);

    // Initialize low-level command message
    InitLowCmd();

    high_state_subscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>>(TOPIC_HIGHSTATE);
    high_state_subscriber_->InitChannel(
        [this](auto&& msg)
        {
            highStateMessageHandler(std::forward<decltype(msg)>(msg));
        }, 1);


    return SystemInterface::on_init(info);
}

/***************************************************************************************/
/*******                                                                         *******/
/*******                             on_configure                                *******/
/*******                                                                         *******/
/***************************************************************************************/
hardware_interface::CallbackReturn Go2HardwareInterface::on_configure(
    const rclcpp_lifecycle::State& previous_state)
{
    RCLCPP_INFO(rclcpp::get_logger(logger_name_), "Configuring Go2 Hardware Interface...");
    // Configuration logic here
    return hardware_interface::CallbackReturn::SUCCESS;
}


/***************************************************************************************/
/*******                                                                         *******/
/*******                              on_activate                                *******/
/*******                                                                         *******/
/***************************************************************************************/
hardware_interface::CallbackReturn Go2HardwareInterface::on_activate(
    const rclcpp_lifecycle::State& previous_state)
{
    RCLCPP_INFO(rclcpp::get_logger(logger_name_), "Activating Go2 Hardware Interface...");
    // Activation logic here
    return hardware_interface::CallbackReturn::SUCCESS;
}


/***************************************************************************************/
/*******                                                                         *******/
/*******                export_state_interfaces (for controller)                 *******/
/*******                                                                         *******/
/***************************************************************************************/
std::vector<hardware_interface::StateInterface> Go2HardwareInterface::export_state_interfaces()
{
    int idx = 0;
    std::vector<hardware_interface::StateInterface> state_interfaces;

    // export joint state interfaces
    for (const auto& joint_name : joint_interface_map_["position"])
    {
        state_interfaces.emplace_back(joint_name, "position", &joint_state_position_[idx++]);
    }

    idx = 0;
    for (const auto& joint_name : joint_interface_map_["velocity"])
    {
        state_interfaces.emplace_back(joint_name, "velocity", &joint_state_velocity_[idx++]);
    }

    idx = 0;
    for (const auto& joint_name : joint_interface_map_["effort"])
    {
        state_interfaces.emplace_back(joint_name, "effort", &joint_state_effort_[idx++]);
    }

    // export imu sensor state interface
    for (uint i = 0; i < info_.sensors[0].state_interfaces.size(); i++)
    {
        state_interfaces.emplace_back(
            info_.sensors[0].name, info_.sensors[0].state_interfaces[i].name, &imu_states_[i]);
    }

    // export foot force sensor state interface
    if (info_.sensors.size() > 1)
    {
        for (uint i = 0; i < info_.sensors[1].state_interfaces.size(); i++)
        {
            state_interfaces.emplace_back(
                info_.sensors[1].name, info_.sensors[1].state_interfaces[i].name, &foot_force_[i]);
        }
    }

    // export odometer state interface
    if (info_.sensors.size() > 2)
    {
        // export high state interface
        for (uint i = 0; i < info_.sensors[2].state_interfaces.size(); i++)
        {
            state_interfaces.emplace_back(
                info_.sensors[2].name, info_.sensors[2].state_interfaces[i].name, &odom_states_[i]);
        }
    }

    return state_interfaces;
}


/***************************************************************************************/
/*******                                                                         *******/
/*******                export_command_interfaces (for controller)               *******/
/*******                                                                         *******/
/***************************************************************************************/
std::vector<hardware_interface::CommandInterface> Go2HardwareInterface::export_command_interfaces()
{
    int idx = 0;
    std::vector<hardware_interface::CommandInterface> command_interfaces;

    for (const auto& joint_name : joint_interface_map_["position"])
    {
        command_interfaces.emplace_back(joint_name, "position", &joint_position_command_[idx++]);
    }

    idx = 0;
    for (const auto& joint_name : joint_interface_map_["velocity"])
    {
        command_interfaces.emplace_back(joint_name, "velocity", &joint_velocity_command_[idx++]);
    }

    idx = 0;
    for (const auto& joint_name : joint_interface_map_["effort"])
    {
        command_interfaces.emplace_back(joint_name, "effort", &joint_torque_command_[idx]);
        command_interfaces.emplace_back(joint_name, "kp", &joint_kp_command_[idx]);
        command_interfaces.emplace_back(joint_name, "kd", &joint_kd_command_[idx]);
        idx++;
    }
    return command_interfaces;
}


/***************************************************************************************/
/*******                                                                         *******/
/*******                                 read                                    *******/
/*******                                                                         *******/
/***************************************************************************************/
hardware_interface::return_type Go2HardwareInterface::read(
    const rclcpp::Time& time, const rclcpp::Duration& period)
{
    // joint states
    for (int i = 0; i < info_.joints.size(); ++i)
    {
        joint_state_position_[i] = robot_low_state_.motor_state()[i].q();
        joint_state_velocity_[i] = robot_low_state_.motor_state()[i].dq();
        joint_state_effort_[i] = robot_low_state_.motor_state()[i].tau_est();
    }

    // imu states
    imu_states_[0] = robot_low_state_.imu_state().quaternion()[0];      // w
    imu_states_[1] = robot_low_state_.imu_state().quaternion()[1];      // x
    imu_states_[2] = robot_low_state_.imu_state().quaternion()[2];      // y
    imu_states_[3] = robot_low_state_.imu_state().quaternion()[3];      // z
    imu_states_[4] = robot_low_state_.imu_state().gyroscope()[0];       // wx
    imu_states_[5] = robot_low_state_.imu_state().gyroscope()[1];       // wy  
    imu_states_[6] = robot_low_state_.imu_state().gyroscope()[2];       // wz
    imu_states_[7] = robot_low_state_.imu_state().accelerometer()[0];   // ax
    imu_states_[8] = robot_low_state_.imu_state().accelerometer()[1];   // ay
    imu_states_[9] = robot_low_state_.imu_state().accelerometer()[2];   // az
    imu_states_[10] = robot_low_state_.imu_state().rpy()[0];            // roll
    imu_states_[11] = robot_low_state_.imu_state().rpy()[1];            // pitch
    imu_states_[12] = robot_low_state_.imu_state().rpy()[2];            // yaw
    imu_states_[13] = robot_low_state_.imu_state().temperature();       // temperature

    // contact states
    foot_force_[0] = robot_low_state_.foot_force()[0];
    foot_force_[1] = robot_low_state_.foot_force()[1];
    foot_force_[2] = robot_low_state_.foot_force()[2];
    foot_force_[3] = robot_low_state_.foot_force()[3];

    if (show_foot_force_)
    {
        RCLCPP_INFO(rclcpp::get_logger(logger_name_), "foot_force(): %f, %f, %f, %f", 
                    foot_force_[0], foot_force_[1], foot_force_[2], foot_force_[3]);
    }

    // odometry states
    odom_states_[0] = robot_high_state_.position()[0];        // px
    odom_states_[1] = robot_high_state_.position()[1];        // py
    odom_states_[2] = robot_high_state_.position()[2];        // pz
    odom_states_[3] = robot_high_state_.velocity()[0];        // vx
    odom_states_[4] = robot_high_state_.velocity()[1];        // vy
    odom_states_[5] = robot_high_state_.velocity()[2];        // vz

    // RCLCPP_INFO(get_logger(), "high state: %f %f %f %f %f %f", high_states_[0], high_states_[1], high_states_[2],
    //             high_states_[3], high_states_[4], high_states_[5]);

    return hardware_interface::return_type::OK;
}


/***************************************************************************************/
/*******                                                                         *******/
/*******                                 write                                   *******/
/*******                                                                         *******/
/***************************************************************************************/
hardware_interface::return_type Go2HardwareInterface::write(
    const rclcpp::Time& time, const rclcpp::Duration& period)
{
    // send motor commands to low-level controller
    for (int i(0); i < 12; ++i)
    {
        robot_low_cmd_.motor_cmd()[i].mode() = 0x01;
        robot_low_cmd_.motor_cmd()[i].q() = static_cast<float>(joint_position_command_[i]);
        robot_low_cmd_.motor_cmd()[i].dq() = static_cast<float>(joint_velocity_command_[i]);
        robot_low_cmd_.motor_cmd()[i].kp() = static_cast<float>(joint_kp_command_[i]);
        robot_low_cmd_.motor_cmd()[i].kd() = static_cast<float>(joint_kd_command_[i]);
        robot_low_cmd_.motor_cmd()[i].tau() = static_cast<float>(joint_torque_command_[i]);
    }

    robot_low_cmd_.crc() = crc32_core(reinterpret_cast<uint32_t*>(&robot_low_cmd_),
                                (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
    low_cmd_publisher_->Write(robot_low_cmd_);
    return hardware_interface::return_type::OK;
}


// From Unitree GO2 SDK Demo
void Go2HardwareInterface::lowStateMessageHandler(const void* messages)
{
    robot_low_state_ = *static_cast<const unitree_go::msg::dds_::LowState_*>(messages);
}

// From Unitree GO2 SDK Demo
void Go2HardwareInterface::highStateMessageHandler(const void* messages)
{
    robot_high_state_ = *static_cast<const unitree_go::msg::dds_::SportModeState_*>(messages);
}


// From Unitree GO2 SDK Demo
void Go2HardwareInterface::InitLowCmd()
{
    robot_low_cmd_.head()[0] = 0xFE;
    robot_low_cmd_.head()[1] = 0xEF;
    robot_low_cmd_.level_flag() = 0xFF;
    robot_low_cmd_.gpio() = 0;

    for (int i = 0; i < 20; i++)
    {
        // motor switch to servo (PMSM) mode
        robot_low_cmd_.motor_cmd()[i].mode() = 0x01;
        robot_low_cmd_.motor_cmd()[i].q() = 0;
        robot_low_cmd_.motor_cmd()[i].kp() = 0;
        robot_low_cmd_.motor_cmd()[i].dq() = 0;
        robot_low_cmd_.motor_cmd()[i].kd() = 0;
        robot_low_cmd_.motor_cmd()[i].tau() = 0;
    }
}

} // namespace edge::go2_hardware_interface

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(edge::go2_hardware_interface::Go2HardwareInterface, 
                       hardware_interface::SystemInterface)