#include <chrono>
#include <random>
#include <vector>
#include <memory>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "edge_msgs/msg/tracking_error.hpp"
#include <rclcpp/rclcpp.hpp>

using namespace std::chrono_literals;

class MockPublisher : public rclcpp::Node {
public:
  MockPublisher()
  : Node("mock_publisher"),
    gen_(rd_()), dist_(-0.1, 0.1)
  {
    publisher_ = this->create_publisher<edge_msgs::msg::TrackingError>("edge/tracking_error", 30);
    timer_ = this->create_wall_timer(100ms, std::bind(&MockPublisher::timer_callback, this));
    RCLCPP_INFO(this->get_logger(), "MockPublisher started.");
  }

private:
void timer_callback() {
  auto msg = edge_msgs::msg::TrackingError();

  for (size_t i = 0; i < msg.tracking_error.size(); i++)
  {
    msg.tracking_error[i] = dist_(gen_);
  }
  publisher_->publish(msg);
  
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < msg.tracking_error.size(); ++i)
  {
      ss << msg.tracking_error[i];
      if (i < msg.tracking_error.size() - 1)
          ss << ", ";
  }
  ss << "]";

  RCLCPP_INFO(this->get_logger(), "Published tracking_error: %s", ss.str().c_str());

}

  rclcpp::Publisher<edge_msgs::msg::TrackingError>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::random_device rd_;
  std::mt19937 gen_;
  std::uniform_real_distribution<double> dist_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MockPublisher>());
  rclcpp::shutdown();
  return 0;
}
