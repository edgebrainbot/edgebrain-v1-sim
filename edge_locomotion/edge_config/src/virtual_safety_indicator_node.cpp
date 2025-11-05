#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <visualization_msgs/msg/marker.hpp>

class VirtualSafetyIndicatorNode : public rclcpp::Node
{
public:
    VirtualSafetyIndicatorNode() : Node("virtual_safety_indicator_node")
    {
        // Create publisher for visualization marker
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/edge/indicator_marker", 10);

        // Subscribe to safety state
        state_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/edge/is_safe", 10,
            std::bind(&VirtualSafetyIndicatorNode::stateCallback, this, std::placeholders::_1));

        // Initialize marker
        marker_.header.frame_id = "base";
        marker_.ns = "virtual_indicator";
        marker_.id = 0;
        marker_.type = visualization_msgs::msg::Marker::SPHERE;
        marker_.action = visualization_msgs::msg::Marker::ADD;
        marker_.pose.position.x = 0.0;
        marker_.pose.position.y = 0.0;
        marker_.pose.position.z = 0.0;
        marker_.scale.x = marker_.scale.y = marker_.scale.z = 0.2;
        marker_.color.a = 1.0;

        RCLCPP_INFO(this->get_logger(), "🟢 Virtual Safety Indicator Node initialized.");
    }

private:
    void stateCallback(const std_msgs::msg::Bool::SharedPtr msg)
    {
        bool is_safe = msg->data;
        marker_.header.stamp = this->now();

        if (is_safe)
        {
            marker_.color.r = 0.0;
            marker_.color.g = 1.0;
            marker_.color.b = 0.0;
        }
        else
        {
            marker_.color.r = 1.0;
            marker_.color.g = 0.0;
            marker_.color.b = 0.0;
        }

        marker_pub_->publish(marker_);
    }

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr state_sub_;
    visualization_msgs::msg::Marker marker_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VirtualSafetyIndicatorNode>());
    rclcpp::shutdown();
    return 0;
}
