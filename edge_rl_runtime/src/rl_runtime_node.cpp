/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <functional>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"

#include "edge_msgs/msg/transition.hpp"
#include "edge_msgs/msg/vec12_msg.hpp"
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
		topic_ns_ = declare_parameter<std::string>("topic_namespace", "edge");
		go2_state_topic_ = declare_parameter<std::string>("go2_state_topic", "/edge/go2/high_state_vec12");
		RCLCPP_INFO(get_logger(), "Use GO2 state topic as robot state source: %s", go2_state_topic_.c_str());

		action_sub_ = create_subscription<edge_msgs::msg::EdgeAction>(
			topic_ns_ + "/terminal_action", 100,
			[this](const edge_msgs::msg::EdgeAction::SharedPtr msg) -> void
			{
				if (msg->action_value.size() < 3) {
					RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
						"Received EdgeAction with action_value size %zu (<3), ignore.",
						msg->action_value.size());
					return;
				}

				curr_action_vel_ = *msg;
				if (curr_action_vel_.action_type != edge_msgs::msg::EdgeAction::CMD_VEL) {
					RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
						"Received EdgeAction with action_type %u (not CMD_VEL), ignore.",
						curr_action_vel_.action_type);
					return;
				} else {
					// Extract cmd_vel action and apply clamping
					latest_vx_cmd_ = clamp(msg->action_value[0], -kMaxVx, kMaxVx);
					latest_vy_cmd_ = clamp(msg->action_value[1], -kMaxVy, kMaxVy);
					latest_wz_cmd_ = clamp(msg->action_value[2], -kMaxWz, kMaxWz);
					has_cmd_ = true;
				}
			});

		// Subscribe to goal point
		goal_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
			"/goal_point", 100,
			[this](const geometry_msgs::msg::PointStamped::SharedPtr msg) -> void
			{
				latest_goal_x_ = msg->point.x;
				latest_goal_y_ = msg->point.y;
			});

		// Subscribe to GO2 high state relayed as ROS Vec12 message
		go2_state_sub_ = create_subscription<edge_msgs::msg::Vec12Msg>(
			go2_state_topic_, 100,
			[this](const edge_msgs::msg::Vec12Msg::SharedPtr msg) -> void
			{
				latest_go2_state_ = edge::ros_interfaces::Vec12MsgToEigen(*msg);
				has_go2_state_ = true;
			});

		// Subscribe to "control input"
		control_input_sub_ = create_subscription<control_input_msgs::msg::Inputs>(
			"/control_input", 100, 
			[this](const control_input_msgs::msg::Inputs::SharedPtr msg) -> void
		{
			switch (msg->command)
			{
				case 6:
					is_started_ = false;
					is_paused_ = true;
					RCLCPP_INFO(get_logger(), "Received pause command.");
					break;
				case 7:
					is_started_ = true;
					is_paused_ = false;
					RCLCPP_INFO(get_logger(), "Received start command.");
					break;
				default:
					RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
						"Received control input with unrecognized command %d, ignoring.",
						msg->command);
					break;
			}
		});
		
		// Publishers
		rclcpp::QoS transition_qos(200);
		transition_qos.best_effort();

		command_action_pub_ = create_publisher<edge_msgs::msg::EdgeAction>(topic_ns_ + "/command_action", 50);
		cmd_vel_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(topic_ns_ + "/cmd_vel", 50);
		transition_pub_ = create_publisher<edge_msgs::msg::Transition>(topic_ns_ + "/transition", transition_qos);	// Best effort for transition to avoid queueing old transitions
		done_pub_ = create_publisher<std_msgs::msg::Bool>(topic_ns_ + "/is_task_done", 20);

		const double dt = std::max(1e-3, 1.0 / kStepHz);
		timer_ = create_wall_timer(std::chrono::duration<double>(dt),
			std::bind(&RLRuntimeNode::stepOnce, this));

		resetEpisode();

		RCLCPP_INFO(get_logger(), "RL runtime node ready. cmd_vel=/edge/cmd_vel transition=/edge/transition");
	}

private:
	using Vector2d = edge::ros_interfaces::Vector2d;
	using Vector3d = edge::ros_interfaces::Vector3d;
	using Vector6d = edge::ros_interfaces::Vector6d;
	using Vector12d = edge::ros_interfaces::Vector12d;

	static constexpr double kStepHz = 50.0;
	static constexpr int kEpisodeHorizonSteps = 12000;
	static constexpr double kGoalX = 5.0;
	static constexpr double kGoalY = 0.0;
	static constexpr double kGoalReachedThreshold = 0.3;
	static constexpr double kObstacleX = 2.5;
	static constexpr double kObstacleY = 0.0;
	static constexpr double kObstacleRadius = 0.4;
	static constexpr double kMaxVx = 1.0;
	static constexpr double kMaxVy = 0.8;
	static constexpr double kMaxWz = 0.5;
	static constexpr double kNominalHeight = 0.33;

	static double clamp(double value, double low, double high)
	{
		return std::min(high, std::max(low, value));
	}

	static double wrapToPi(double angle)
	{
		constexpr double kPi = 3.14159265358979323846;
		while (angle > kPi) {
			angle -= 2.0 * kPi;
		}
		while (angle < -kPi) {
			angle += 2.0 * kPi;
		}
		return angle;
	}

	void resetEpisode()
	{
		x_ = 0.0;
		y_ = 0.0;
		yaw_ = 0.0;
		step_count_ = 0;
		last_action_vel_ = edge_msgs::msg::EdgeAction();
		curr_action_vel_ = edge_msgs::msg::EdgeAction();
		last_action_vel_.action_type = edge_msgs::msg::EdgeAction::CMD_VEL;
		last_action_vel_.action_mode = edge_msgs::msg::EdgeAction::TEACHER;
		last_action_vel_.action_value = {0.0, 0.0, 0.0};
		curr_action_vel_ = last_action_vel_;
		done_ = false;
		last_robot_state_ = getRobotState();
		curr_robot_state_ = last_robot_state_;
	}

	Vector12d getRobotState() const
	{
		if (has_go2_state_) {
			return latest_go2_state_;
		}

		Vector12d robot_state = Vector12d::Zero();
		robot_state(0) = x_;
		robot_state(1) = y_;
		robot_state(2) = kNominalHeight;
		robot_state(5) = yaw_;
		robot_state(6) = latest_vx_cmd_;
		robot_state(7) = latest_vy_cmd_;
		robot_state(11) = latest_wz_cmd_;
		return robot_state;
	}

	Vector12d getDRLStudentGoalState() const
	{
		Vector12d goal_state = Vector12d::Zero();
		goal_state(0) = latest_goal_x_;
		goal_state(1) = latest_goal_y_;
		goal_state(2) = kNominalHeight;
		return goal_state;
	}

	// obs
	std::vector<double> getObservation() const
	{
		return edge::ros_interfaces::Vec12MsgToVector(edge::ros_interfaces::EigenToVec12Msg(last_robot_state_));
	}

	// obs_next
	std::vector<double> getObservationNext() const
	{
		return edge::ros_interfaces::Vec12MsgToVector(edge::ros_interfaces::EigenToVec12Msg(curr_robot_state_));
	}

	// action
	edge_msgs::msg::EdgeAction getDRLAction() const
	{
		return last_action_vel_;
	}

	double getObstacleMinDistance() const
	{
		const double dx = x_ - kObstacleX;
		const double dy = y_ - kObstacleY;
		return std::max(1e-3, std::hypot(dx, dy) - kObstacleRadius);
	}

	bool isGoalReached() const
	{
		return std::hypot(latest_goal_x_ - x_, latest_goal_y_ - y_) < kGoalReachedThreshold;
	}

	bool processTransition()
	{
		auto safe_action_value = [this](const edge_msgs::msg::EdgeAction & action, size_t idx) {
			if (idx < action.action_value.size()) {
				return action.action_value[idx];
			}
			RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
				"last_action_vel.action_value size is %zu, expect >=3. Use 0 for missing entries.",
				action.action_value.size());
			return 0.0;
		};

		Vector2d nav_goal;
		nav_goal << latest_goal_x_, latest_goal_y_;

		Vector3d robot_pos;
		robot_pos << curr_robot_state_(0), curr_robot_state_(1), curr_robot_state_(5);

		// Reward factors
		const bool is_fall = false;
		const bool goal_reached = isGoalReached();
		const bool timeout = step_count_ >= kEpisodeHorizonSteps;
		const Vector3d curr_vel_action = (Vector3d() << latest_vx_cmd_, latest_vy_cmd_, latest_wz_cmd_).finished();
		const Vector3d last_vel_action =
			(Vector3d() << safe_action_value(last_action_vel_, 0),
			 safe_action_value(last_action_vel_, 1),
			 safe_action_value(last_action_vel_, 2)).finished();

		edge::config::RewardFactor reward_factors{
			{"is_teacher_activated", false},
			{"time_elapsed", static_cast<double>(step_count_)},
			{"obstacle_distance_min", getObstacleMinDistance()},
			{"nav_goal", nav_goal},
			{"robot_pos", robot_pos},
			{"robot_state", curr_robot_state_},
			{"is_fall", is_fall},
			{"is_goal_reached", goal_reached}
		};

		// Compute reward components with error handling
		std::vector<edge_msgs::msg::EdgeReward> reward_components;
		try {
			reward_components = reward_composer_.computeTotalReward(reward_factors);
		} catch (const std::exception & e) {
			RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
				"Reward compose failed (check reward YAML keys/factors): %s", e.what());
		}

		// Transition message
		edge_msgs::msg::Transition transition;
		transition.header.stamp = now();
		transition.obs = getObservation();
		transition.action = getDRLAction();
		transition.reward_components = std::move(reward_components);
		transition.next_obs = getObservationNext();
		transition.done = goal_reached || timeout || is_fall;

		// Publish transition
		transition_pub_->publish(transition);

		return transition.done;
	}

	void stepOnce()
	{
		if (done_) {
			resetEpisode();
		}

		if (!has_cmd_) {
			RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
				"Waiting for action command on topic %s/terminal_action...", topic_ns_.c_str());
			return;
		}

		if (!is_started_) {
			RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
				"Waiting for start command on /control_input (command=4)...");
			return;
		}

		RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
			"Step %d: Processing transition...", step_count_);


		step_count_ += 1;

		// Obtain the latest robot state from GO2 state topic
		curr_robot_state_ = getRobotState();

		// Process transition
		done_ = processTransition();

		// Publish done signal
		std_msgs::msg::Bool done_msg;
		done_msg.data = done_;
		done_pub_->publish(done_msg);

		// Publish the terminal action to executor
		command_action_pub_->publish(curr_action_vel_);

		// Update last action and last robot state for next step
		last_action_vel_ = curr_action_vel_;
		last_robot_state_ = curr_robot_state_;
	}

	// Reward composer
	edge::config::RewardComposer reward_composer_{""};
	std::string topic_ns_;
	std::string go2_state_topic_{"/edge/go2/high_state_vec12"};

	bool has_cmd_{false};
	bool has_go2_state_{false};
	bool done_{false};
	int step_count_{0};

	double latest_vx_cmd_{0.0};
	double latest_vy_cmd_{0.0};
	double latest_wz_cmd_{0.0};
	double latest_goal_x_{kGoalX};
	double latest_goal_y_{kGoalY};

	double x_{0.0};
	double y_{0.0};
	double yaw_{0.0};

	bool is_started_{false};
	bool is_paused_{false};

	Vector12d last_robot_state_ = Vector12d::Zero();
	Vector12d curr_robot_state_ = Vector12d::Zero();
	Vector12d latest_go2_state_ = Vector12d::Zero();

	// Vector3d last_action_vel_ = Vector3d::Zero();
	// Vector3d curr_action_vel_ = Vector3d::Zero();
	edge_msgs::msg::EdgeAction last_action_vel_;
	edge_msgs::msg::EdgeAction curr_action_vel_;

	// Subscribers
	rclcpp::Subscription<edge_msgs::msg::EdgeAction>::SharedPtr action_sub_;
	rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr goal_sub_;
	rclcpp::Subscription<edge_msgs::msg::Vec12Msg>::SharedPtr go2_state_sub_;
	rclcpp::Subscription<control_input_msgs::msg::Inputs>::SharedPtr control_input_sub_;

	// Publishers
	rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_pub_;
	rclcpp::Publisher<edge_msgs::msg::Transition>::SharedPtr transition_pub_;
	rclcpp::Publisher<edge_msgs::msg::EdgeAction>::SharedPtr command_action_pub_;
	rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr done_pub_;
	rclcpp::TimerBase::SharedPtr timer_;
};
}	// namespace edge::rl_runtime

int main(int argc, char ** argv)
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<edge::rl_runtime::RLRuntimeNode>());
	rclcpp::shutdown();
	return 0;
}
