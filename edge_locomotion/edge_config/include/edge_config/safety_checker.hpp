#ifndef EDGE_CONFIG_SAFETY_CHECKER_HPP
#define EDGE_CONFIG_SAFETY_CHECKER_HPP

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <vector>
#include <string>

namespace edge::config
{

struct SafetyLimits
{
  double roll_limit_rad;
  double pitch_limit_rad;
  double height_limit_m;
};

class SafetyChecker
{
public:
  explicit SafetyChecker(double roll_deg, double pitch_deg, double height_m);

  bool checkSafety(const std::vector<double> errorState);

private:
  bool checkSafetySet(double height, double roll, double pitch);
  void safetyStatusLogging();

  SafetyLimits limits_;
  bool is_curr_safe_{true};
  bool is_last_safe_{true};
  std::vector<std::string> violation_reasons_;
};

}  // namespace edge

#endif  // EDGE_CONFIG_SAFETY_CHECKER_HPP
