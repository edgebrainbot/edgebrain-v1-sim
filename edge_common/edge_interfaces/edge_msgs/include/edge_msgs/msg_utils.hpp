#pragma once
#include <cstdint>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <builtin_interfaces/msg/time.hpp>

namespace edge::utils {

enum class ActionMode : uint8_t {
    STUDENT = 0,
    TEACHER = 1,
    UNCERTAIN = 2,
};

inline uint8_t ToMsg(ActionMode mode) {
    return static_cast<uint8_t>(mode);
}

inline ActionMode FromMsg(uint8_t value) {
    return (value == 1) ? ActionMode::TEACHER : ActionMode::STUDENT;
}

inline std::string ToString(ActionMode mode) {
    switch (mode) {
        case ActionMode::STUDENT: return "STUDENT";
        case ActionMode::TEACHER: return "TEACHER";
        default: return "UNCERTAIN";
    }
}

inline ActionMode FromInt(uint8_t value) {
    return (value == 1) ? ActionMode::TEACHER : ActionMode::STUDENT;
}

inline builtin_interfaces::msg::Time getCurrentROSTime() {
  // Use a static Clock to avoid creating a new instance repeatedly
  static rclcpp::Clock clock(RCL_ROS_TIME);
  return clock.now();
}

}  // namespace edge::utils
