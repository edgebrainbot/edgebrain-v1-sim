#ifndef SAFETY_MONITOR_HPP
#define SAFETY_MONITOR_HPP

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <unordered_map>
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

class SafetyMonitor
{
public:
  explicit SafetyMonitor(double roll_deg, double pitch_deg, double height_m);

  bool checkSafety(const std::vector<double> errorState);

private:
  bool checkSafetySet(double height, double roll, double pitch);
  void safetyStatusLogging();

  SafetyLimits limits_;
  bool is_curr_safe_{true};
  bool is_last_safe_{true};
  std::vector<std::string> violation_reasons_;
};

}  // namespace edge::config

#endif  // SAFETY_MONITOR_HPP
