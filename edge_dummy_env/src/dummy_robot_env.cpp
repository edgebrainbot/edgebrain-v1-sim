/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "std_msgs/msg/bool.hpp"

#include "edge_msgs/msg/transition.hpp"
#include "edge_msgs/msg/edge_vel_action.hpp"
#include "edge_config/reward_composer.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"

class DummyRobotEnvNode : public rclcpp::Node
{
public:
	DummyRobotEnvNode()
	: Node("dummy_robot_env"),
		reward_composer_(declare_parameter<std::string>("reward_yaml_path", ""))
	{
		topic_ns_ = declare_parameter<std::string>("topic_namespace", "edge");

		action_sub_ = create_subscription<edge_msgs::msg::EdgeVelAction>(
			topic_ns_ + "/terminal_action_vel", 100,
			[this](const edge_msgs::msg::EdgeVelAction::SharedPtr msg) -> void
			{
				latest_vx_cmd_ = clamp(msg->cmd_vel_action[0], -kMaxVx, kMaxVx);
				latest_vy_cmd_ = clamp(msg->cmd_vel_action[1], -kMaxVy, kMaxVy);
				latest_wz_cmd_ = clamp(msg->cmd_vel_action[2], -kMaxWz, kMaxWz);
				has_cmd_ = true;
			});

		cmd_vel_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
			"/cmd_vel",
			20,
			std::bind(&DummyRobotEnvNode::cmdVelCallback, this, std::placeholders::_1));

		transition_pub_ = create_publisher<edge_msgs::msg::Transition>(topic_ns_ + "/transition", 20);
		done_pub_ = create_publisher<std_msgs::msg::Bool>(topic_ns_ + "/is_task_done", 20);

		const double dt = std::max(1e-3, 1.0 / kStepHz);
		timer_ = create_wall_timer(
			std::chrono::duration<double>(dt),
			std::bind(&DummyRobotEnvNode::stepOnce, this));

		resetEpisode();

		RCLCPP_INFO(get_logger(), "Dummy env ready. cmd_vel=/cmd_vel transition=/edge_dummy_env/transition");
	}

private:
	using Vector2d = edge::ros_interfaces::Vector2d;
	using Vector3d = edge::ros_interfaces::Vector3d;
	using Vector6d = edge::ros_interfaces::Vector6d;
	using Vector12d = edge::ros_interfaces::Vector12d;

	static constexpr double kStepHz = 50.0;
	static constexpr int kEpisodeHorizonSteps = 1200;
	static constexpr double kGoalX = 5.0;
	static constexpr double kGoalY = 0.0;
	static constexpr double kGoalReachedThreshold = 0.3;
	static constexpr double kObstacleX = 2.5;
	static constexpr double kObstacleY = 0.0;
	static constexpr double kObstacleRadius = 0.4;
	static constexpr double kMaxVx = 1.0;
	static constexpr double kMaxVy = 0.8;
	static constexpr double kMaxWz = 1.5;
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

	static edge_msgs::msg::EdgeAction toEdgeActionMsg(double vx, double vy, double wz)
	{
		edge_msgs::msg::EdgeAction action;
		action.control_action[0] = vx;
		action.control_action[1] = vy;
		action.control_action[2] = 0.0;
		action.control_action[3] = 0.0;
		action.control_action[4] = 0.0;
		action.control_action[5] = wz;
		action.action_mode = 0;
		return action;
	}

	void cmdVelCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
	{
		latest_vx_cmd_ = clamp(msg->twist.linear.x, -kMaxVx, kMaxVx);
		latest_vy_cmd_ = clamp(msg->twist.linear.y, -kMaxVy, kMaxVy);
		latest_wz_cmd_ = clamp(msg->twist.angular.z, -kMaxWz, kMaxWz);
		has_cmd_ = true;
	}

	void resetEpisode()
	{
		x_ = 0.0;
		y_ = 0.0;
		yaw_ = 0.0;
		step_count_ = 0;
		last_action_ = Vector6d::Zero();
		done_ = false;
	}

	std::vector<double> getObservation() const
	{
		Vector12d obs = Vector12d::Zero();
		obs(0) = kGoalX - x_;
		obs(1) = kGoalY - y_;
		obs(2) = 0.0;
		obs(3) = 0.0;
		obs(4) = 0.0;
		obs(5) = wrapToPi(-yaw_);
		obs(6) = latest_vx_cmd_;
		obs(7) = latest_vy_cmd_;
		obs(8) = 0.0;
		obs(9) = 0.0;
		obs(10) = 0.0;
		obs(11) = latest_wz_cmd_;
		return edge::ros_interfaces::EigenToVector(obs);
	}

	double getObstacleMinDistance() const
	{
		const double dx = x_ - kObstacleX;
		const double dy = y_ - kObstacleY;
		return std::max(1e-3, std::hypot(dx, dy) - kObstacleRadius);
	}

	bool isGoalReached() const
	{
		return std::hypot(kGoalX - x_, kGoalY - y_) < kGoalReachedThreshold;
	}

	void integrateDynamics(double dt)
	{
		const double cos_yaw = std::cos(yaw_);
		const double sin_yaw = std::sin(yaw_);

		const double vx_world = latest_vx_cmd_ * cos_yaw - latest_vy_cmd_ * sin_yaw;
		const double vy_world = latest_vx_cmd_ * sin_yaw + latest_vy_cmd_ * cos_yaw;

		x_ += vx_world * dt;
		y_ += vy_world * dt;
		yaw_ = wrapToPi(yaw_ + latest_wz_cmd_ * dt);
	}

	void stepOnce()
	{
		if (done_) {
			resetEpisode();
		}

		if (!has_cmd_) {
			return;
		}

		const double dt = std::max(1e-3, 1.0 / kStepHz);
		std::vector<double> obs = getObservation();

		const Vector6d curr_action = (Vector6d() << latest_vx_cmd_, latest_vy_cmd_, 0.0, 0.0, 0.0, latest_wz_cmd_).finished();
		const Vector6d delta_action = curr_action - last_action_;
		last_action_ = curr_action;

		integrateDynamics(dt);
		step_count_ += 1;

		Vector12d robot_state = Vector12d::Zero();
		robot_state(0) = x_;
		robot_state(1) = y_;
		robot_state(2) = kNominalHeight;
		robot_state(5) = yaw_;
		robot_state(6) = latest_vx_cmd_;
		robot_state(7) = latest_vy_cmd_;
		robot_state(11) = latest_wz_cmd_;

		Vector2d nav_goal;
		nav_goal << kGoalX, kGoalY;

		Vector3d robot_pos;
		robot_pos << x_, y_, yaw_;

		const bool goal_reached = isGoalReached();
		const bool timeout = step_count_ >= kEpisodeHorizonSteps;
		const bool is_fall = false;

		edge::config::RewardFactor reward_factors{
			{"is_teacher_activated", false},
			{"time_elapsed", static_cast<double>(step_count_)},
			{"obstacle_distance_min", getObstacleMinDistance()},
			{"nav_goal", nav_goal},
			{"robot_pos", robot_pos},
			{"robot_state", robot_state},
			{"delta_action", delta_action},
			{"is_fall", is_fall},
			{"is_goal_reached", goal_reached}
		};

		std::vector<edge_msgs::msg::EdgeReward> reward_components;
		try {
			reward_components = reward_composer_.computeTotalReward(reward_factors);
		} catch (const std::exception & e) {
			RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
				"Reward compose failed (check reward YAML keys/factors): %s", e.what());
		}

		edge_msgs::msg::Transition transition;
		transition.header.stamp = now();
		transition.obs = std::move(obs);
		transition.action = toEdgeActionMsg(latest_vx_cmd_, latest_vy_cmd_, latest_wz_cmd_);
		transition.reward_components = std::move(reward_components);
		transition.next_obs = getObservation();
		transition.done = goal_reached || timeout || is_fall;

		transition_pub_->publish(transition);

		std_msgs::msg::Bool done_msg;
		done_msg.data = transition.done;
		done_pub_->publish(done_msg);

		done_ = transition.done;
	}

	edge::config::RewardComposer reward_composer_;
	std::string topic_ns_;

	bool has_cmd_{false};
	bool done_{false};
	int step_count_{0};

	double latest_vx_cmd_{0.0};
	double latest_vy_cmd_{0.0};
	double latest_wz_cmd_{0.0};

	double x_{0.0};
	double y_{0.0};
	double yaw_{0.0};

	Vector6d last_action_ = Vector6d::Zero();

	rclcpp::Subscription<edge_msgs::msg::EdgeVelAction>::SharedPtr action_sub_;
	rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_sub_;
	rclcpp::Publisher<edge_msgs::msg::Transition>::SharedPtr transition_pub_;
	rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr done_pub_;
	rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<DummyRobotEnvNode>());
	rclcpp::shutdown();
	return 0;
}
