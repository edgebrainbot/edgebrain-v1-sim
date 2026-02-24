/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ament_index_cpp/get_package_share_directory.hpp>
#include "edge_quadruped_controller/edge_quadruped_controller.hpp"

namespace edge::control::quadruped
{
    using config_type = controller_interface::interface_configuration_type;

    /***************************************************************************************/
    /*******                                                                         *******/
    /*******                      state_interface_configuration                      *******/
    /*******                                                                         *******/
    /***************************************************************************************/
    controller_interface::InterfaceConfiguration EdgeQuadrupedController::state_interface_configuration() const
    {
        controller_interface::InterfaceConfiguration config = {config_type::INDIVIDUAL, {}};

        config.names.reserve(joint_names_.size() * state_interface_types_.size());
        for (const auto& joint_name : joint_names_)
        {
            for (const auto& interface_type : state_interface_types_)
            {
                config.names.push_back(joint_name + "/" + interface_type);
            }
        }

        for (const auto& interface_type : imu_interface_types_)
        {
            config.names.push_back(imu_name_ + "/" + interface_type);
        }
        return config;
    }

    /***************************************************************************************/
    /*******                                                                         *******/
    /*******                     command_interface_configuration                     *******/
    /*******                                                                         *******/
    /***************************************************************************************/
    controller_interface::InterfaceConfiguration EdgeQuadrupedController::command_interface_configuration() const
    {
        controller_interface::InterfaceConfiguration config = {config_type::INDIVIDUAL, {}};

        config.names.reserve(joint_names_.size() * command_interface_types_.size());
        for (const auto& joint_name : joint_names_)
        {
            for (const auto& interface_type : command_interface_types_)
            {
                if (!command_prefix_.empty())
                {
                    config.names.push_back(command_prefix_ + "/" + joint_name + "/" + interface_type);
                }
                else
                {
                    config.names.push_back(joint_name + "/" + interface_type);
                }
            }
        }
        return config;
    }

    /***************************************************************************************/
    /*******                                                                         *******/
    /*******                                 on_init                                 *******/
    /*******                                                                         *******/
    /***************************************************************************************/
    controller_interface::CallbackReturn EdgeQuadrupedController::on_init()
    {
        try
        {
            // Topic namespace
            topic_ns_ = auto_declare<std::string>("topic_namespace", "edge");

            // Frame name
            odom_frame_id_ = auto_declare<std::string>("odom_frame_id", "odom");
            base_frame_id_ = auto_declare<std::string>("base_frame_id", "base_link");

            // Joint names and interface types
            joint_names_ = auto_declare<std::vector<std::string>>("joints", joint_names_);
            state_interface_types_ = auto_declare<std::vector<std::string>>("state_interfaces", state_interface_types_);
            command_interface_types_ = auto_declare<std::vector<std::string>>("command_interfaces", command_interface_types_);

            // Robot parameters - use robot_name to auto-generate package names
            robot_name_ = auto_declare<std::string>("robot_name", "unitree_go2");

            // imu sensor
            imu_name_ = auto_declare<std::string>("imu_name", imu_name_);
            imu_interface_types_ = auto_declare<std::vector<std::string>>("imu_interfaces", state_interface_types_);
            command_prefix_ = auto_declare<std::string>("command_prefix", command_prefix_);

            // pose parameters
            down_pos_ = auto_declare<std::vector<double>>("down_pos", down_pos_);
            stand_pos_ = auto_declare<std::vector<double>>("stand_pos", stand_pos_);
            stand_gains_ = auto_declare<std::vector<double>>("stand_gains", stand_gains_);
            swing_gains_ = auto_declare<std::vector<double>>("swing_gains", swing_gains_);
            stable_gains_ = auto_declare<std::vector<double>>("stable_gains", stable_gains_);

            // Gait parameters 
            gait_height_ = auto_declare<double>("gait_height", gait_height_);
            RCLCPP_INFO(get_node()->get_logger(), "Gait Height: %.3f m", gait_height_);

            // Controller manager update rate
            get_node()->get_parameter("update_rate", ctrl_interfaces_.frequency_);
            RCLCPP_INFO(get_node()->get_logger(), "Controller Manager Update Rate: %d Hz", ctrl_interfaces_.frequency_);

            // Initialize robot model with auto-generated URDF path
            initializeRobotModel();

            // Reset service parameters
            reset_cooldown_ = auto_declare<double>("reset_cooldown", 1.0);
            RCLCPP_INFO(get_node()->get_logger(), "Reset Cooldown: %.2f s", reset_cooldown_);

            // Initialize reset time
            last_reset_timestep_ = 0;

            ctrl_component_.estimator_ = std::make_shared<Estimator>(ctrl_interfaces_, ctrl_component_, 
                                                                     odom_frame_id_, base_frame_id_, get_node());
            ctrl_component_.balance_ctrl_ = std::make_shared<BalanceCtrl>(ctrl_component_.robot_model_);

            ctrl_component_.phase_scheduler_ = std::make_shared<PhaseScheduler>(
                0.45, 0.5, Vec4(0, 0.5, 0.5, 0), get_node()->now());
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
            return controller_interface::CallbackReturn::ERROR;
        }
        return CallbackReturn::SUCCESS;
    }


    /***************************************************************************************/
    /*******                                                                         *******/
    /*******                               on_configure                              *******/
    /*******                                                                         *******/
    /***************************************************************************************/
    controller_interface::CallbackReturn EdgeQuadrupedController::on_configure(
        const rclcpp_lifecycle::State& /*previous_state*/)
    {
        try {
            // Load RESIDUAL_GAIN from physical_model.yaml (already loaded via --param-file)
            // Check if parameter exists and get it
            std::vector<double> residual_gain_vec;
            if (get_node()->has_parameter("RESIDUAL_GAIN")) {
                residual_gain_vec = get_node()->get_parameter("RESIDUAL_GAIN").as_double_array();
            } else {
                RCLCPP_WARN(get_node()->get_logger(), "RESIDUAL_GAIN parameter not found, using zero matrix");
                residual_gain_vec.clear();
            }
            
            // Validate parameter size (should be 72 for 6x12 matrix)
            if (residual_gain_vec.empty()) {
                RCLCPP_WARN(get_node()->get_logger(), "RESIDUAL_GAIN parameter is empty, using zero matrix");
                ctrl_component_.residual_gain_ = Eigen::MatrixXd::Zero(6, 12);
            } else if (residual_gain_vec.size() != 72) {
                RCLCPP_ERROR(get_node()->get_logger(), "RESIDUAL_GAIN size mismatch: expected 72, got %zu", residual_gain_vec.size());
                return controller_interface::CallbackReturn::ERROR;
            } else {
                // Convert to Eigen matrix (6x12) and store in ctrl_component_
                ctrl_component_.residual_gain_ = Eigen::MatrixXd(6, 12);
                for (int i = 0; i < 6; i++) {
                    for (int j = 0; j < 12; j++) {
                        ctrl_component_.residual_gain_(i, j) = residual_gain_vec[i * 12 + j];
                        RCLCPP_DEBUG(get_node()->get_logger(), "RESIDUAL_GAIN[%d][%d] = %.6f", i, j, ctrl_component_.residual_gain_(i, j));
                    }
                }
                RCLCPP_INFO(get_node()->get_logger(), "Loaded RESIDUAL_GAIN matrix (6x12) from parameter file");
                std::cout << "Loaded RESIDUAL_GAIN matrix (6x12):\n" << ctrl_component_.residual_gain_ << std::endl;    

            }

            // Subscribe to "control input"
            control_input_sub_ = get_node()->create_subscription<control_input_msgs::msg::Inputs>(
                "/control_input", 100, 
                [this](const control_input_msgs::msg::Inputs::SharedPtr msg) -> void
            {
                // Handle message
                ctrl_interfaces_.control_inputs_.command = msg->command;
                ctrl_interfaces_.control_inputs_.lx = msg->lx;
                ctrl_interfaces_.control_inputs_.ly = msg->ly;
                ctrl_interfaces_.control_inputs_.rx = msg->rx;
                ctrl_interfaces_.control_inputs_.ry = msg->ry;
            });

            // Subscribe to "cmd_vel"
            cmd_sub_ = get_node()->create_subscription<geometry_msgs::msg::TwistStamped>(
                "/cmd_vel", 100,
                [this](const geometry_msgs::msg::TwistStamped::SharedPtr msg) -> void
            {
                cmd_vx_ = msg->twist.linear.x;
                cmd_vy_ = msg->twist.linear.y;
                cmd_wz_ = msg->twist.angular.z;
                // std::cout << "*************************************************************" << std::endl;
                // std::cout << "Received /cmd_vel: "
                //         << "vx = " << cmd_vx_ << ", "
                //         << "vy = " << cmd_vy_ << ", "
                //         << "wz = " << cmd_wz_ << std::endl;
            });

            // Subscribe to "edge/terminal_action"
            action_sub_ = get_node()->create_subscription<edge_msgs::msg::EdgeAction>(
                topic_ns_ + "/terminal_action", 100,
                [this](const edge_msgs::msg::EdgeAction::SharedPtr msg) -> void
            {
                latest_edge_action_ = *msg;
                // std::cout << "Received new action from the trigger." << std::endl;
                // for (int i = 0; i < 6; i++) {
                //     std::cout << "accel[" << i << "] = " << latest_edge_action_.control_action[i] << std::endl;
                // }
            });

            // Subscribe to fall status
            is_fall_sub_ = get_node()->create_subscription<std_msgs::msg::Bool>(
                topic_ns_ + "/is_fall", 100,
                [this](const std_msgs::msg::Bool::SharedPtr msg) -> void
            {
                is_fall_ = msg->data;
            });

            // Subscribe to goal point
            goal_sub_ = get_node()->create_subscription<geometry_msgs::msg::PointStamped>(
                "/goal_point", 100,
                [this](const geometry_msgs::msg::PointStamped::SharedPtr msg) -> void
            {
                latest_goal_x_ = msg->point.x;
                latest_goal_y_ = msg->point.y;
                // std::cout << "Received new goal from the trigger." << std::endl;
                // std::cout << "latest_edge_goal x: " << latest_goal_x_ << std::endl;
                // std::cout << "latest_edge_goal y: " << latest_goal_y_ << std::endl;
            });

            // Subscribe to obstacle average distance
            obstacle_avg_distance_sub_ = get_node()->create_subscription<std_msgs::msg::Float32>(
                "/obstacle_avg_distance", 100,
                [this](const std_msgs::msg::Float32::SharedPtr msg) -> void
            {
                obstacle_avg_distance_ = msg->data;
                // std::cout << "Received new obstacle average distance." << std::endl;
                // std::cout << "obstacle_avg_distance: " << obstacle_avg_distance_ << std::endl;
            });

            // Subscribe to obstacle minimum distance
            obstacle_min_distance_sub_ = get_node()->create_subscription<std_msgs::msg::Float32>(
                "/obstacle_min_distance", 100,
                [this](const std_msgs::msg::Float32::SharedPtr msg) -> void
            {
                obstacle_min_distance_ = msg->data;
            });

            // Subscribe to far_reach_goal_status to update is_goal_reached_
            is_goal_reached_sub_ = get_node()->create_subscription<std_msgs::msg::Bool>(
                "/far_reach_goal_status", 100,
                [this](const std_msgs::msg::Bool::SharedPtr msg) -> void
            {
                is_goal_reached_ = msg->data;
                // std::cout << "Received new is_goal_reached." << std::endl;
                // std::cout << "is_goal_reached: " << is_goal_reached_ << std::endl;
            });

            // Reset service
            setupResetService();

            // Launch service
            setupLaunchService();

            // Publisher for tracking error
            tracking_error_tea_pub_ = get_node()->create_publisher<edge_msgs::msg::Vec12Msg>(
                topic_ns_ + "/tracking_error", 100);

            // Publisher for tracking error of DRL student
            tracking_error_stu_pub_ = get_node()->create_publisher<edge_msgs::msg::Vec12Msg>(
                topic_ns_ + "/tracking_error_stu", 100);

            // Publisher for robot state
            robot_state_pub_ = get_node()->create_publisher<edge_msgs::msg::Vec12Msg>(
                topic_ns_ + "/robot_state", 100);   

            // Publisher for experience
            transition_pub_ = get_node()->create_publisher<edge_msgs::msg::Transition>(
                topic_ns_ + "/transition", 100);

            // Publisher for is_task_done
            is_task_done_pub_ = get_node()->create_publisher<std_msgs::msg::Bool>(
                topic_ns_ + "/is_task_done", 100);

        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_node()->get_logger(), "Failed to create subscription: %s", e.what());
            throw;
        }

        return CallbackReturn::SUCCESS;
    }

    /***************************************************************************************/
    /*******                                                                         *******/
    /*******                               on_activate                               *******/
    /*******                                                                         *******/
    /***************************************************************************************/
    controller_interface::CallbackReturn EdgeQuadrupedController::on_activate(
        const rclcpp_lifecycle::State& /*previous_state*/)
    {
        // clear out vectors in case of restart
        ctrl_interfaces_.clear();

        // assign command interfaces
        for (auto& interface : command_interfaces_)
        {
            std::string interface_name = interface.get_interface_name();
            if (const size_t pos = interface_name.find('/'); pos != std::string::npos)
            {
                command_interface_map_[interface_name.substr(pos + 1)]->push_back(interface);
            }
            else
            {
                command_interface_map_[interface_name]->push_back(interface);
            }
        }

        // assign state interfaces
        for (auto& interface : state_interfaces_)
        {
            if (interface.get_prefix_name() == imu_name_)
            {
                ctrl_interfaces_.imu_state_interface_.emplace_back(interface);
            }
            else
            {
                state_interface_map_[interface.get_interface_name()]->push_back(interface);
            }
        }

        // Create FSM List
        state_list_.passive = std::make_shared<FSMPassive>(ctrl_interfaces_);
        state_list_.fixedDown = std::make_shared<FSMFixedDown>(ctrl_interfaces_, down_pos_, stand_gains_[0], stand_gains_[1]);
        state_list_.fixedStand = std::make_shared<FSMFixedStand>(ctrl_interfaces_, ctrl_component_, stand_pos_, stand_gains_[0], stand_gains_[1]);
        state_list_.freeStand = std::make_shared<FSMFreeStand>(ctrl_interfaces_, ctrl_component_, stand_gains_[0], stand_gains_[1]);
        state_list_.balanceTest = std::make_shared<FSMBalanceTest>(ctrl_interfaces_, ctrl_component_, stable_gains_);
        state_list_.trotting = std::make_shared<FSMTrotting>(ctrl_interfaces_, ctrl_component_, swing_gains_, stable_gains_, gait_height_);
        
        // Initialize FSM
        current_state_ = state_list_.passive;
        current_state_->enter();
        next_state_ = current_state_;
        next_state_name_ = current_state_->state_name;
        mode_ = FSMMode::NORMAL;

        return CallbackReturn::SUCCESS;
    }

    /***************************************************************************************/
    /*******                                                                         *******/
    /*******                                update                                   *******/
    /*******                                                                         *******/
    /***************************************************************************************/
    controller_interface::return_type EdgeQuadrupedController::update(const rclcpp::Time& time, const rclcpp::Duration& period)
    {
        ctrl_component_.robot_model_->update();
        ctrl_component_.phase_scheduler_->update(time);
        ctrl_component_.estimator_->update();

        if (mode_ == FSMMode::NORMAL)
        {
            if (auto trot = std::dynamic_pointer_cast<FSMTrotting>(current_state_)) {
                trot->run(time, period, 
                          latest_edge_action_, 
                          tracking_error_tea_pub_, 
                          tracking_error_stu_pub_,
                          robot_state_pub_,
                          transition_pub_, 
                          is_task_done_pub_,
                          cmd_vx_, cmd_vy_, cmd_wz_, latest_goal_x_, latest_goal_y_,
                          obstacle_avg_distance_,
                          obstacle_min_distance_,
                          is_goal_reached_,
                          is_fall_);
                          
            } else {
                current_state_->run(time, period);
            }
            
            next_state_name_ = current_state_->checkChange();
            if (next_state_name_ != current_state_->state_name)
            {
                mode_ = FSMMode::CHANGE;
                next_state_ = getNextState(next_state_name_);
                RCLCPP_INFO(get_node()->get_logger(), "Switched from %s to %s",
                            current_state_->state_name_string.c_str(), next_state_->state_name_string.c_str());

                // If switching to trotting state, set last reset time
                if (auto trot = std::dynamic_pointer_cast<FSMTrotting>(next_state_)) {
                    trot->setLastResetTime(rclcpp::Clock().now());
                }
            }
        }
        else if (mode_ == FSMMode::CHANGE)
        {
            current_state_->exit();
            current_state_ = next_state_;

            // Set PhaseScheduler status based on FSM state
            if (ctrl_component_.phase_scheduler_)
            {
                ctrl_component_.phase_scheduler_->setStatusFromFSM(next_state_name_);
            }

            current_state_->enter();
            mode_ = FSMMode::NORMAL;
        }

        return controller_interface::return_type::OK;
    }

    /***************************************************************************************/
    /*******                                                                         *******/
    /*******                              on_deactivate                              *******/
    /*******                                                                         *******/
    /***************************************************************************************/
    controller_interface::CallbackReturn EdgeQuadrupedController::on_deactivate(
        const rclcpp_lifecycle::State& /*previous_state*/)
    {
        release_interfaces();
        return CallbackReturn::SUCCESS;
    }

    /***************************************************************************************/
    /*******                                                                         *******/
    /*******                                on_cleanup                               *******/
    /*******                                                                         *******/
    /***************************************************************************************/
    controller_interface::CallbackReturn
    EdgeQuadrupedController::on_cleanup(const rclcpp_lifecycle::State& /*previous_state*/)
    {
        return CallbackReturn::SUCCESS;
    }

    /***************************************************************************************/
    /*******                                                                         *******/
    /*******                                on_error                                 *******/
    /*******                                                                         *******/
    /***************************************************************************************/
    controller_interface::CallbackReturn
    EdgeQuadrupedController::on_error(const rclcpp_lifecycle::State& /*previous_state*/)
    {
        return CallbackReturn::ERROR;
    }

    /***************************************************************************************/
    /*******                                                                         *******/
    /*******                               on_shutdown                               *******/
    /*******                                                                         *******/
    /***************************************************************************************/
    controller_interface::CallbackReturn
    EdgeQuadrupedController::on_shutdown(const rclcpp_lifecycle::State& /*previous_state*/)
    {
        return CallbackReturn::SUCCESS;
    }

    void EdgeQuadrupedController::initializeRobotModel()
    {
        try
        {
            // Automatically build file paths and initialize interface
            const std::string robot_pkg = robot_name_ + "_description";
            const std::string config_path = ament_index_cpp::get_package_share_directory(robot_pkg);

            std::string urdf_file = config_path + "/urdf/" + "robot.urdf";

            Vec12 stand_joint_positions;
            for (int i = 0; i < 4; ++i)
            {
                stand_joint_positions.segment(3 * i, 3) = Eigen::Map<const Vec3>(&stand_pos_[3 * i]);
            }

            // Initialize robot model with default stand joint positions
            ctrl_component_.robot_model_ = std::make_shared<QuadrupedModel>(
                ctrl_interfaces_, urdf_file, joint_names_, stand_joint_positions);

            RCLCPP_INFO(get_node()->get_logger(), "Robot model initialized successfully for robot: %s",
                        robot_name_.c_str());
            RCLCPP_INFO(get_node()->get_logger(), "URDF file: %s", urdf_file.c_str());
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(get_node()->get_logger(), "Failed to initialize robot model: %s", e.what());
            throw;
        }

    }

    std::shared_ptr<FSMState> EdgeQuadrupedController::getNextState(const FSMStateName stateName) const
    {
        switch (stateName)
        {
        case FSMStateName::INVALID:
            return state_list_.invalid;
        case FSMStateName::PASSIVE:
            return state_list_.passive;
        case FSMStateName::FIXEDDOWN:
            return state_list_.fixedDown;
        case FSMStateName::FIXEDSTAND:
            return state_list_.fixedStand;
        case FSMStateName::FREESTAND:
            return state_list_.freeStand;
        case FSMStateName::TROTTING:
            return state_list_.trotting;
        case FSMStateName::BALANCETEST:
            return state_list_.balanceTest;
        default:
            return state_list_.invalid;
        }
    }

    void EdgeQuadrupedController::setupResetService()
    {
        reset_service_ = get_node()->create_service<std_srvs::srv::Trigger>(
            topic_ns_ + "/reset/controller",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                std::shared_ptr<std_srvs::srv::Trigger::Response> response)
            {
                // Debounce check
                auto now = get_node()->now();
                // double time_since_last = (now - last_reset_time_).seconds();
                
                // if (time_since_last < reset_cooldown_) {
                //     response->success = false;
                //     response->message = "Reset cooldown active (" + 
                //         std::to_string(reset_cooldown_ - time_since_last) + "s remaining)";
                //     RCLCPP_WARN(get_node()->get_logger(), "%s", response->message.c_str());
                //     return;
                // }
                
                // Execute reset
                try {
                    // Reset estimator
                    if (ctrl_component_.estimator_) {
                        ctrl_component_.estimator_->reset();
                    }
                    
                    // Reset gait generator if in trotting state
                    if (auto trot = std::dynamic_pointer_cast<FSMTrotting>(current_state_)) {
                        trot->getGaitGenerator().reset();
                        trot->setReset(true);       // inform the trotting state of the reset
                    }
                    
                    // Update timestamp
                    // last_reset_time_ = now;
                    response->success = true;
                    response->message = "Reset successful";
                    
                    RCLCPP_INFO(get_node()->get_logger(), 
                        "Reset triggered by service call at time %.3f", now.seconds());
                        
                } catch (const std::exception& e) {
                    response->success = false;
                    response->message = std::string("Reset failed: ") + e.what();
                    RCLCPP_ERROR(get_node()->get_logger(), "Reset failed: %s", e.what());
                }
            }
        );
    
        RCLCPP_INFO(get_node()->get_logger(), 
            "Reset service created at: %s/reset/controller (cooldown: %.2fs)", 
            topic_ns_.c_str(), reset_cooldown_);
    }

    void EdgeQuadrupedController::setupLaunchService()
    {
        launch_service_ = get_node()->create_service<std_srvs::srv::Trigger>(
            topic_ns_ + "/launch/controller",
            [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                std::shared_ptr<std_srvs::srv::Trigger::Response> response)
            {
                // Debounce check
                auto now = get_node()->now();
                // double time_since_last = (now - last_reset_time_).seconds();
    
                // Execute reset
                try {
                    current_state_->setLaunchAfterReset(true);
                    
                    // Update timestamp
                    // last_reset_time_ = now;
                    response->success = true;
                    response->message = "Launch successful";
                    
                    RCLCPP_INFO(get_node()->get_logger(), 
                        "Launch triggered by service call at time %.3f", now.seconds());
                        
                } catch (const std::exception& e) {
                    response->success = false;
                    response->message = std::string("Launch failed: ") + e.what();
                    RCLCPP_ERROR(get_node()->get_logger(), "Launch failed: %s", e.what());
                }
            }
        );
    
        RCLCPP_INFO(get_node()->get_logger(), 
            "Launch service created at: %s/controller/launch (cooldown: %.2fs)", 
            topic_ns_.c_str(), reset_cooldown_);
    }
}



#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(edge::control::quadruped::EdgeQuadrupedController, controller_interface::ControllerInterface);
