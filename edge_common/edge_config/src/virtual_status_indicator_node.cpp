#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <edge_msgs/msg/edge_action.hpp>
#include <visualization_msgs/msg/marker.hpp>

struct RGBColor
{
    float r;
    float g;
    float b;
    float a;  // alpha
};

std_msgs::msg::ColorRGBA toColorRGBA(const RGBColor &c)
{
    std_msgs::msg::ColorRGBA color;
    color.r = c.r;
    color.g = c.g;
    color.b = c.b;
    color.a = c.a;
    return color;
}

std::map<std::string, RGBColor> color_map = {
    {"GREEN",   {0.0f, 1.0f, 0.0f, 1.0f}},
    {"RED",   {1.0f, 0.0f, 0.0f, 1.0f}}, 
    {"BLUE",     {0.0f, 0.0f, 1.0f, 1.0f}},  
    {"YELLOW",  {1.0f, 1.0f, 0.0f, 1.0f}}, 
    {"CYAN",    {0.0f, 1.0f, 1.0f, 1.0f}}, 
    {"MAGENTA", {1.0f, 0.0f, 1.0f, 1.0f}}, 
    {"WHITE",   {1.0f, 1.0f, 1.0f, 1.0f}}, 
    {"BLACK",   {0.0f, 0.0f, 0.0f, 1.0f}},
    {"GRAY",    {0.5f, 0.5f, 0.5f, 1.0f}},
    {"ORANGE",  {1.0f, 0.65f, 0.0f, 1.0f}},
    {"PURPLE",  {0.5f, 0.0f, 0.5f, 1.0f}},
    {"PINK",    {1.0f, 0.75f, 0.8f, 1.0f}},
    {"BROWN",   {0.6f, 0.4f, 0.2f, 1.0f}},
    {"DARK_GREEN", {0.0f, 0.39f, 0.0f, 1.0f}},
    {"LIGHT_BLUE", {0.68f, 0.85f, 0.9f, 1.0f}},
    {"DARK_RED",   {0.55f, 0.0f, 0.0f, 1.0f}},
    {"LIGHT_GREEN", {0.56f, 0.93f, 0.56f, 1.0f}},
    {"DARK_BLUE",  {0.0f, 0.0f, 0.55f, 1.0f}},
    {"LIGHT_YELLOW", {1.0f, 1.0f, 0.88f, 1.0f}},
    {"TRANSPARENT", {0.0f, 0.0f, 0.0f, 0.0f}},
};

class VirtualStatusIndicatorNode : public rclcpp::Node
{
public:
    VirtualStatusIndicatorNode() : Node("virtual_status_indicator_node")
    {
        // Create publisher for visualization marker
        safe_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/edge/safe_indicator_marker", 10);
        
        // Create publisher for action mode marker
        action_mode_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/edge/action_mode_indicator_marker", 10);

        // Subscribe to safety state
        safe_status_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/edge/is_safe", 10,
            std::bind(&VirtualStatusIndicatorNode::safeStatusCallback, this, std::placeholders::_1));

        // Subscribe to action mode
        action_mode_sub_ = this->create_subscription<edge_msgs::msg::EdgeAction>(
            "/edge/terminal_action", 10,
            std::bind(&VirtualStatusIndicatorNode::actionModeCallback, this, std::placeholders::_1));

        // Initialize safe_marker
        safe_marker_.header.frame_id = "base_link";
        safe_marker_.ns = "virtual_status_indicator";
        safe_marker_.id = 0;
        safe_marker_.type = visualization_msgs::msg::Marker::SPHERE;
        safe_marker_.action = visualization_msgs::msg::Marker::ADD;
        safe_marker_.pose.position.x = 0.0;
        safe_marker_.pose.position.y = 0.0;
        safe_marker_.pose.position.z = 0.0;
        safe_marker_.scale.x = safe_marker_.scale.y = safe_marker_.scale.z = 0.2;
        safe_marker_.color.a = 1.0;

        // Initialize action_mode_marker
        action_mode_marker_.header.frame_id = "base_link";
        action_mode_marker_.ns = "virtual_status_indicator";
        action_mode_marker_.id = 0;
        action_mode_marker_.type = visualization_msgs::msg::Marker::SPHERE;
        action_mode_marker_.action = visualization_msgs::msg::Marker::ADD;
        action_mode_marker_.pose.position.x = 0.0;
        action_mode_marker_.pose.position.y = 0.0;
        action_mode_marker_.pose.position.z = 0.0;
        action_mode_marker_.scale.x = action_mode_marker_.scale.y = action_mode_marker_.scale.z = 0.2;
        action_mode_marker_.color.a = 1.0;

        RCLCPP_INFO(this->get_logger(), "🟢 Virtual Status Indicator Node initialized.");
    }

private:
    void safeStatusCallback(const std_msgs::msg::Bool::SharedPtr msg)
    {
        bool is_safe = msg->data;
        safe_marker_.header.stamp = this->now();

        if (is_safe) {
            safe_marker_.color =  toColorRGBA(color_map["GREEN"]);
        }
        else {
            safe_marker_.color = toColorRGBA(color_map["RED"]);
        }

        safe_marker_pub_->publish(safe_marker_);
    }

    void actionModeCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg)
    {
        int action_mode = msg->action_mode;     // 0 -- student, 1 -- teacher
        action_mode_marker_.header.stamp = this->now();

        if (action_mode) {       // teacher mode
            action_mode_marker_.color = toColorRGBA(color_map["YELLOW"]);
        }
        else {      // student mode
            action_mode_marker_.color = toColorRGBA(color_map["BLUE"]);
        }

        action_mode_marker_pub_->publish(action_mode_marker_);
    }

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr safe_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr action_mode_marker_pub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr safe_status_sub_;
    rclcpp::Subscription<edge_msgs::msg::EdgeAction>::SharedPtr action_mode_sub_;
    visualization_msgs::msg::Marker safe_marker_;
    visualization_msgs::msg::Marker action_mode_marker_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VirtualStatusIndicatorNode>());
    rclcpp::shutdown();
    return 0;
}
