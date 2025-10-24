#include <chrono>
#include <random>
#include <vector>
#include <memory>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <rclcpp/rclcpp.hpp>

using namespace std::chrono_literals;

class MockPublisher : public rclcpp::Node {
public:
  MockPublisher()
  : Node("mock_publisher"),
    gen_(rd_()), dist_(-0.1, 0.1)
  {
    publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("tracking_error", 30);
    timer_ = this->create_wall_timer(100ms, std::bind(&MockPublisher::timer_callback, this));
    RCLCPP_INFO(this->get_logger(), "MockPublisher started.");
  }

private:
void timer_callback() {
  std_msgs::msg::Float64MultiArray msg;
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < 10; ++i) {
    msg.data.push_back(dist_(gen_));
    oss << msg.data[i];
    if (i < 9) oss << ", ";
  }
  oss << "]";
  publisher_->publish(msg);
  RCLCPP_INFO(this->get_logger(), "Published tracking_error: %s", oss.str().c_str());
}

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
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
