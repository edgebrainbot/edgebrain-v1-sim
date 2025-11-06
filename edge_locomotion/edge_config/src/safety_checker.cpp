#include <cmath>
#include <rclcpp/rclcpp.hpp>
#include "edge_config/safety_checker.hpp"
#include <std_msgs/msg/float64_multi_array.hpp>

namespace edge::config
{

SafetyChecker::SafetyChecker(double roll_deg, double pitch_deg, double height_m)
{
  // === Convert to internal limits ===
  limits_.roll_limit_rad  = roll_deg * M_PI / 180.0;
  limits_.pitch_limit_rad = pitch_deg * M_PI / 180.0;
  limits_.height_limit_m  = height_m;

  std::cout << "--- 🚨 SafetyChecker initialized with ---" << std::endl;
  std::cout << "  roll_limit_deg  = " << roll_deg << "°" << std::endl;
  std::cout << "  pitch_limit_deg = " << pitch_deg << "°" << std::endl;
  std::cout << "  height_limit_m  = " << height_m << " m" << std::endl;

  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "  roll_limit_rad  = " << limits_.roll_limit_rad << " rad" << std::endl;
  std::cout << "  pitch_limit_rad = " << limits_.pitch_limit_rad << " rad" << std::endl;
  std::cout << "  height_limit_m  = " << limits_.height_limit_m << " m" << std::endl;
}

// --------------------------------------------------------
bool SafetyChecker::checkSafety(const std::vector<double> errorState)
{
  if (errorState.size() < 3) {
    std::cerr << "Invalid errorState size!" << std::endl;
    return false;
  }

  double height = errorState[2];
  double roll   = errorState[3];
  double pitch  = errorState[4];

  is_curr_safe_ = checkSafetySet(height, roll, pitch);
  safetyStatusLogging();

  is_last_safe_ = is_curr_safe_;
  violation_reasons_.clear();

  return is_curr_safe_;
}

// --------------------------------------------------------
bool SafetyChecker::checkSafetySet(double height, double roll, double pitch)
{
  bool is_safe = true;

  if (std::abs(roll) > limits_.roll_limit_rad) {
    is_safe = false;
    violation_reasons_.push_back("roll = " + std::to_string(roll * 180.0 / M_PI) + "°");
  }

  if (std::abs(pitch) > limits_.pitch_limit_rad) {
    is_safe = false;
    violation_reasons_.push_back("pitch = " + std::to_string(pitch * 180.0 / M_PI) + "°");
  }

  if (std::abs(height) > limits_.height_limit_m) {
    is_safe = false;
    violation_reasons_.push_back("height = " + std::to_string(height) + "m");
  }

  return is_safe;
}

// --------------------------------------------------------
void SafetyChecker::safetyStatusLogging()
{
  // Logging
  if (!is_curr_safe_) {
    std::string msg = "⚠️ Current System is UNSAFE! Reasons: ";
    for (auto &r : violation_reasons_) msg += r + " | ";

    if (is_last_safe_)
        msg += "System leaving the learning space! Safety Triggered.";
    else
        msg += "System still outside the learning space.";

    std::cerr << msg << std::endl;
  } else {
    if (!is_last_safe_)
      std::cout << "✅ System returned to the learning space." << std::endl;
    else
      std::cout << "✅ System remains within the learning space." << std::endl;
  }
}

}  // namespace edge
