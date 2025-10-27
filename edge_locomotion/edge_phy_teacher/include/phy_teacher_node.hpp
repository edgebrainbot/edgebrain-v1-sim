#ifndef PHY_TEACHER_NODE_HPP
#define PHY_TEACHER_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/float64_multi_array.hpp"
#include "geometry_msgs/msg/accel.hpp"
#include "lmi_optimizer.hpp"
#include "edge_config/safety_checker.hpp"

namespace edge::phy_teacher {

class PhyTeacherNode : public rclcpp::Node {
public:

  explicit PhyTeacherNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~PhyTeacherNode() override;

private:
  void errorCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void lmiCallback();

  // For phy-teacher logistics
  std::vector<double> last_error_state_;   // Last received tracking error
  std::vector<double> patch_center_;      // Center of the safety patch
  bool teacher_active_ = false;            // Whether the teacher is active
  double chi_ = 0.0;                        // Hyperparameter χ
  bool got_msg_ = false;
  bool print_process_time_ = false;
  lmi::LMIOptimizer lmi_optimizer_;

  // Safety-conditions
  double roll_limit_rad_;
  double pitch_limit_rad_;
  double height_limit_;
  bool is_last_safe_ = true;
  std::shared_ptr<edge::SafetyChecker> safety_checker_;

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_;
  // rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_;
  rclcpp::Publisher<geometry_msgs::msg::Accel>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr lmi_timer_;
};

}  // namespace edge

#endif  // PHY_TEACHER_NODE_HPP