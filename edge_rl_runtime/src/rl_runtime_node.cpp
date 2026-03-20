/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <algorithm>
#include <functional>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "edge_msgs/msg/transition.hpp"
#include "edge_msgs/msg/edge_state.hpp"
#include "edge_config/reward_composer.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"
#include "control_input_msgs/msg/inputs.hpp"

namespace edge::rl_runtime
{

class RLRuntimeNode : public rclcpp::Node
{
public:

    RLRuntimeNode() : Node("rl_runtime")
    {
        state_input_topic_ = declare_parameter<std::string>("state_input_topic", "/edge/state_input");
        goal_reached_threshold_ = declare_parameter<double>("navigation.goal_threshold", 0.5);
        episode_horizon_steps_ = declare_parameter<int>("episode.horizon_steps", 120000);

        RCLCPP_INFO(get_logger(), "Use state input topic: %s", state_input_topic_.c_str());
        RCLCPP_INFO(get_logger(), "Use goal done threshold: %.3f", goal_reached_threshold_);
        RCLCPP_INFO(get_logger(), "Use episode horizon steps: %d", episode_horizon_steps_);

        /* ACTION SUBSCRIBER
         * action arrival triggers environment step */
        action_sub_ = create_subscription<edge_msgs::msg::EdgeAction>(
            "terminal_action",
            100,
            std::bind(&RLRuntimeNode::actionCallback, this, std::placeholders::_1)
        );

        // Subscribe to obstacle minimum distance
        obstacle_min_distance_sub_ = create_subscription<std_msgs::msg::Float32>(
            "/obstacle_min_distance", 100,
            [this](const std_msgs::msg::Float32::SharedPtr msg) -> void
        {
            obstacle_min_distance_ = msg->data;
        });

        /* Goal */
        goal_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
            "/goal_point",
            100,
            [this](const geometry_msgs::msg::PointStamped::SharedPtr msg)
            {
                latest_goal_x_ = msg->point.x;
                latest_goal_y_ = msg->point.y;
            });

        /* State Input */
        state_input_sub_ = create_subscription<edge_msgs::msg::EdgeState>(
            state_input_topic_,
            100,
            [this](const edge_msgs::msg::EdgeState::SharedPtr msg)
            {
                curr_edge_state_ = *msg;
                has_state_input_ = true;
            });

        /* Control Input */
        control_input_sub_ = create_subscription<control_input_msgs::msg::Inputs>(
            "/control_input",
            100,
            std::bind(&RLRuntimeNode::controlInputCallback, this, std::placeholders::_1)
        );

        /* Odometry */
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odom",
            100,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg)
            {
                curr_odom_ = *msg;
            }
        );

        /* Publishers */
        rclcpp::QoS transition_qos(200);
        transition_qos.best_effort();

        command_action_pub_ = create_publisher<edge_msgs::msg::EdgeAction>("command_action", 50);
        transition_pub_ = create_publisher<edge_msgs::msg::Transition>("transition", transition_qos);
        done_pub_ = create_publisher<std_msgs::msg::Bool>("is_task_done", 20);
        is_real_robot_ready_pub_ = create_publisher<std_msgs::msg::Bool>("/edge/is_real_robot_ready", 10);
        distance_to_goal_pub_ = create_publisher<std_msgs::msg::Float32>("/edge/distance_to_goal_debug", 10);

        reset();

        RCLCPP_INFO(get_logger(), "RL runtime ready (action-driven)");
    }

private:

    using Vector2d = edge::ros_interfaces::Vector2d;
    using Vector3d = edge::ros_interfaces::Vector3d;
    using Vector12d = edge::ros_interfaces::Vector12d;

    static double clamp(double v,double lo,double hi)
    {
        return std::min(hi,std::max(lo,v));
    }

    /*
     * ACTION CALLBACK
     * action arrival triggers step()
     */

    void actionCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg)
    {
        if(msg->action_value.size() < 3) {
            RCLCPP_WARN(get_logger(),"action size < 3");
            return;
        }

        curr_edge_action_ = *msg;
        has_cmd_ = true;

        if(!is_started_) return;

        stepOnce();
    }

    /* CONTROL INPUT */
    void controlInputCallback(const control_input_msgs::msg::Inputs::SharedPtr msg)
    {
        switch(msg->command)
        {
            case 6:
                is_started_ = false;
                is_paused_ = true;
                break;

            case 4:
                is_started_ = true;
                is_paused_ = false;
                break;

            default:
                break;
        }
    }

    // obs
	std::vector<double> getObservation() const
	{
        std::vector<double> obs(last_edge_state_.state_value.begin(), last_edge_state_.state_value.end());
        if (obs.size() >= 2) {
            obs[0] = latest_goal_x_ - last_odom_.pose.pose.position.x;      // use last_odom_ to align with last_edge_state_
            obs[1] = latest_goal_y_ - last_odom_.pose.pose.position.y;
        }
        return obs;
	}

	// obs_next
	std::vector<double> getObservationNext() const
	{
        std::vector<double> obs_next(curr_edge_state_.state_value.begin(), curr_edge_state_.state_value.end());
        if (obs_next.size() >= 2) {
            obs_next[0] = latest_goal_x_ - curr_odom_.pose.pose.position.x;  // use curr_odom_ to align with curr_edge_state_
            obs_next[1] = latest_goal_y_ - curr_odom_.pose.pose.position.y;
        }
        return obs_next;
	}

	// action
	edge_msgs::msg::EdgeAction getDRLAction() const
	{
		return last_edge_action_;
	}

    // is_goal_reached
    bool is_goal_reached() const
    {
        double goal_distance = std::hypot(latest_goal_x_ - curr_odom_.pose.pose.position.x,
                                            latest_goal_y_ - curr_odom_.pose.pose.position.y);
        return goal_distance < goal_reached_threshold_;
    }

    /*
     * STEP FUNCTION
     */

    void stepOnce()
    {
        // if(done_)
        // {
        //     reset();
        // }

        // whether action command is published
        if (!has_cmd_) {
			RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                "Waiting for action command on topic terminal_action...");
			return;
		}

        // whether started
		if (!is_started_) {
			RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
				"Waiting for start command on /control_input (command=4)...");
			return;
		}

		RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Step %d: Processing transition...", step_count_);

        step_count_++;

        // Publish real robot ready signal at the beginning of the episode
        std_msgs::msg::Bool ready_msg;
        ready_msg.data = true;
        is_real_robot_ready_pub_->publish(ready_msg);

        // Process transition and compute reward
        done_ = processTransition();

        // Publish whether episode is done
        std_msgs::msg::Bool done_msg;
        done_msg.data = done_;
        done_pub_->publish(done_msg);

        if(done_)
        {
            done_flag_ = true;  // set done_flag_ to true to send stop action and skip transition publish in the next step
            RCLCPP_INFO(get_logger(), "Episode done at step %d!", step_count_);
        }

        if(done_flag_)
        {
            edge_msgs::msg::EdgeAction stop_action = curr_edge_action_;
            stop_action.action_value = {0.0, 0.0, 0.0};
            command_action_pub_->publish(stop_action);
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                "Episode done: skip transition publish and send stop action [0, 0, 0]");
        }
        else
        {
            command_action_pub_->publish(curr_edge_action_);
        }

        last_edge_action_ = curr_edge_action_;
        last_edge_state_ = curr_edge_state_;
        last_odom_ = curr_odom_;
    }

    /* Transition Processing */
    bool processTransition()
    {
        Vector2d nav_goal;
        nav_goal << latest_goal_x_, latest_goal_y_;

        Vector3d robot_pos;
        Vector12d robot_state;

        if(curr_edge_state_.state_type == edge_msgs::msg::EdgeState::ROBOT_STATE)
        {
            robot_pos <<  curr_odom_.pose.pose.position.x,  // x
                          curr_odom_.pose.pose.position.y,  // y
                          curr_edge_state_.state_value[5];  // yaw

            robot_state << latest_goal_x_ - curr_odom_.pose.pose.position.x,   // delta_x
                           latest_goal_y_ - curr_odom_.pose.pose.position.y,  // delta_y
                           curr_edge_state_.state_value[2],  // z
                           curr_edge_state_.state_value[3],  // roll
                           curr_edge_state_.state_value[4],  // pitch
                           curr_edge_state_.state_value[5],  // yaw
                           curr_edge_state_.state_value[6],  // vx
                           curr_edge_state_.state_value[7],  // vy
                           curr_edge_state_.state_value[8],  // vz
                           curr_edge_state_.state_value[9],  // wx
                           curr_edge_state_.state_value[10], // wy
                           curr_edge_state_.state_value[11]; // wz
        }


        // double goal_distance = std::hypot(latest_goal_x_ - curr_edge_state_.state_value[0],
        //                                   latest_goal_y_ - curr_edge_state_.state_value[1]);

        double goal_distance = std::hypot(latest_goal_x_ - curr_odom_.pose.pose.position.x,
                                            latest_goal_y_ - curr_odom_.pose.pose.position.y);
        auto goal_distance_msg = std::make_unique<std_msgs::msg::Float32>();
        goal_distance_msg->data = static_cast<float>(goal_distance);
        distance_to_goal_pub_->publish(std::move(goal_distance_msg));

        bool goal_reach_flag = is_goal_reached();

        bool timeout = step_count_ >= episode_horizon_steps_;

        edge::config::RewardFactor reward_factors
        {
            {"nav_goal", nav_goal},
            {"robot_pos", robot_pos},
            {"robot_state", robot_state},
            {"is_goal_reached", goal_reach_flag},
            {"time_elapsed", (double)step_count_},
            {"is_teacher_activated", curr_edge_action_.action_mode == 1 ? true : false},
            {"obstacle_distance_min", obstacle_min_distance_}
        };

        std::vector<edge_msgs::msg::EdgeReward> reward_components;

        try
        {
            reward_components =
                reward_composer_.computeTotalReward(reward_factors);
        }
        catch(...)
        {
        }

        const bool is_done = goal_reach_flag || timeout;

        // Publish transition
        edge_msgs::msg::Transition transition;

        transition.header.stamp = now();
        transition.obs = getObservation();
        transition.next_obs = getObservationNext();
        transition.action = last_edge_action_;
        transition.reward_components = std::move(reward_components);
        transition.done = is_done;

        transition_pub_->publish(transition);

        return is_done;
    }

    // Reset the environment to the initial state
    void reset()
    {
        step_count_ = 0;
        done_ = false;
        // is_started_ = false;
        // is_paused_ = true;

        last_edge_state_.state_value.assign(12, 0.0);
        curr_edge_state_.state_value.assign(12, 0.0);
    }

private:

    edge::config::RewardComposer reward_composer_{""};

    std::string state_input_topic_;
    int episode_horizon_steps_{120000};
    double goal_reached_threshold_{0.3};

    bool has_cmd_{false};
    bool has_state_input_{false};
    bool done_{false};
    bool done_flag_{false};

    int step_count_{0};

    bool is_started_{false};
    bool is_paused_{false};

    double latest_goal_x_{0.0};
    double latest_goal_y_{0.0};

    double obstacle_min_distance_{0.0};

    edge_msgs::msg::EdgeState last_edge_state_;
    edge_msgs::msg::EdgeState curr_edge_state_;
    Vector12d latest_state_input_ = Vector12d::Zero();
    nav_msgs::msg::Odometry last_odom_;
    nav_msgs::msg::Odometry curr_odom_;

    edge_msgs::msg::EdgeAction last_edge_action_;
    edge_msgs::msg::EdgeAction curr_edge_action_;

    // Subscribers
    rclcpp::Subscription<edge_msgs::msg::EdgeAction>::SharedPtr action_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr goal_sub_;
    rclcpp::Subscription<edge_msgs::msg::EdgeState>::SharedPtr state_input_sub_;
    rclcpp::Subscription<control_input_msgs::msg::Inputs>::SharedPtr control_input_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr obstacle_min_distance_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    // Publishers
    rclcpp::Publisher<edge_msgs::msg::EdgeAction>::SharedPtr command_action_pub_;
    rclcpp::Publisher<edge_msgs::msg::Transition>::SharedPtr transition_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr is_real_robot_ready_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr done_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr distance_to_goal_pub_;
};

}

int main(int argc,char **argv)
{
    rclcpp::init(argc,argv);
    rclcpp::spin(std::make_shared<edge::rl_runtime::RLRuntimeNode>());
    rclcpp::shutdown();
}