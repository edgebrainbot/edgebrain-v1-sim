#include <chrono>
#include <exception>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "edge_msgs/msg/vec12_msg.hpp"
#include "edge_rl_runtime/go2_high_state_subscriber.hpp"

class Go2HighStateBridgeNode : public rclcpp::Node
{
public:
    Go2HighStateBridgeNode()
    : Node("go2_high_state_bridge")
    {
        network_interface_ = declare_parameter<std::string>("go2_network_interface", "eno1");
        state_topic_ = declare_parameter<std::string>("state_topic", "/edge/go2/high_state_vec12");
        publish_hz_ = declare_parameter<double>("publish_hz", 500.0);

        state_pub_ = create_publisher<edge_msgs::msg::Vec12Msg>(state_topic_, 500);

        try {
            go2_subscriber_ = std::make_unique<edge::rl_runtime::Go2HighStateSubscriber>(network_interface_);
            go2_subscriber_->Init();
            bridge_ready_ = true;
        } catch (const std::exception & e) {
            bridge_ready_ = false;
            RCLCPP_ERROR(get_logger(),
                        "Failed to initialize GO2 high state subscriber (nic=%s): %s",
                        network_interface_.c_str(), e.what());
        } catch (...) {
            bridge_ready_ = false;
            RCLCPP_ERROR(get_logger(),
                        "Failed to initialize GO2 high state subscriber (nic=%s): unknown exception",
                        network_interface_.c_str());
        }

        const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, publish_hz_));
        timer_ = create_wall_timer(period, [this]() {
            if (!bridge_ready_) {
                return;
            }
            if (!go2_subscriber_->HasState()) {
                return;
            }
            state_pub_->publish(go2_subscriber_->GetLatestVec12Msg());
        });

        RCLCPP_INFO(get_logger(), "GO2 high state bridge started. nic=%s topic=%s hz=%.1f",
                    network_interface_.c_str(), state_topic_.c_str(), publish_hz_);
    }

private:
    std::string network_interface_;
    std::string state_topic_;
    double publish_hz_{500.0};
    bool bridge_ready_{false};

    std::unique_ptr<edge::rl_runtime::Go2HighStateSubscriber> go2_subscriber_;
    rclcpp::Publisher<edge_msgs::msg::Vec12Msg>::SharedPtr state_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<Go2HighStateBridgeNode>());
    } catch (const std::exception & e) {
        RCLCPP_ERROR(rclcpp::get_logger("go2_high_state_bridge"),
                    "Unhandled exception: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(rclcpp::get_logger("go2_high_state_bridge"),
                    "Unhandled unknown exception");
    }
    rclcpp::shutdown();
    return 0;
}
