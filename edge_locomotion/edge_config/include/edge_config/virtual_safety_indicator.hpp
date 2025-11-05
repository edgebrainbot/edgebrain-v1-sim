#pragma once

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>

class VirtualSafetyIndicator
{
public:
    explicit VirtualSafetyIndicator(rclcpp::Node::SharedPtr node,
                             const std::string &topic_name = "indicator_marker",
                             const std::string &frame_id = "base_link")
    : node_(node), frame_id_(frame_id)
    {
        pub_marker_ = node_->create_publisher<visualization_msgs::msg::Marker>(topic_name, 10);
        RCLCPP_INFO(node_->get_logger(), "🟢 SafetyIndicator initialized (topic: %s)", topic_name.c_str());
    }

    void update(bool is_safe)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "virtual_safety_indicator";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = 0.0;
        marker.pose.position.y = 0.0;
        marker.pose.position.z = 0.5;
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;

        if (is_safe)
        {
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
        }
        else
        {
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
        }
        marker.color.a = 1.0;

        pub_marker_->publish(marker);
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_marker_;
    std::string frame_id_;
};
