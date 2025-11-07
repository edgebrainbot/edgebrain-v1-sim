#pragma once
/**
 * @file conversions.hpp
 * @brief Unified conversion utilities between Eigen, std::vector, and ROS 2 message types.
 *
 * This header provides high-performance inline conversions between:
 *  - Eigen matrices/vectors (for control/optimization)
 *  - std::vector<double> (for parameters, YAML, serialization)
 *  - ROS message types (Action, TrackingError)
 *
 * The goal is to provide a consistent interface for real-time controllers
 * and runtime-learning nodes while maintaining header-only efficiency.
 */

#include <Eigen/Dense>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>

#include "edge_msgs/msg/action.hpp"
#include "edge_msgs/msg/tracking_error.hpp"

namespace edge::ros_interfaces {

// ======================================================
// ==================== Type Aliases ====================
// ======================================================

using Vector6d  = Eigen::Matrix<double, 6, 1>;
using Vector12d = Eigen::Matrix<double, 12, 1>;
using Mat6x6    = Eigen::Matrix<double, 6, 6>;

// ======================================================
// =============== Eigen ↔ std::vector ==================
// ======================================================

/**
 * @brief Convert any Eigen matrix or vector to a std::vector<double>.
 */
template <typename Derived>
inline std::vector<double> EigenToVector(const Eigen::MatrixBase<Derived>& m) {
  return std::vector<double>(m.derived().data(), m.derived().data() + m.size());
}

/**
 * @brief Convert std::vector<double> to Eigen::VectorXd (dynamic size).
 */
inline Eigen::VectorXd VectorToEigenDynamic(const std::vector<double>& vec) {
  return Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size());
}

/**
 * @brief Convert std::vector<double> to fixed-size Eigen matrix or vector.
 *
 * Example:
 * ```cpp
 * auto v = VectorToEigen<6>(vec);     // 6x1
 * auto M = VectorToEigen<6,6>(vec);   // 6x6
 * ```
 *
 * @tparam Rows  Number of rows
 * @tparam Cols  Number of columns (default 1)
 * @throws std::runtime_error if vec.size() != Rows * Cols
 */
template <int Rows, int Cols = 1>
inline Eigen::Matrix<double, Rows, Cols> VectorToEigen(const std::vector<double>& vec) {
  const size_t expected = static_cast<size_t>(Rows * Cols);
  if (vec.size() != expected) {
    throw std::runtime_error(
      "VectorToEigen: size mismatch, expected " + std::to_string(expected) +
      " elements but got " + std::to_string(vec.size()) + ".");
  }
  return Eigen::Map<const Eigen::Matrix<double, Rows, Cols>>(vec.data());
}

// ======================================================
// =============== std::vector ↔ ROS msg ================
// ======================================================

/**
 * @brief Convert std::vector<double> to edge_msgs::msg::Action.
 */
inline edge_msgs::msg::Action VectorToActionMsg(const std::vector<double>& data) {
  edge_msgs::msg::Action msg;
  const size_t n = std::min(msg.action.size(), data.size());
  std::copy_n(data.begin(), n, msg.action.begin());
  return msg;
}

/**
 * @brief Convert edge_msgs::msg::Action to std::vector<double>.
 */
inline std::vector<double> ActionMsgToVector(const edge_msgs::msg::Action& msg) {
  return std::vector<double>(msg.action.begin(), msg.action.end());
}

/**
 * @brief Convert std::vector<double> to edge_msgs::msg::TrackingError.
 */
inline edge_msgs::msg::TrackingError VectorToTrackingErrorMsg(const std::vector<double>& data) {
  edge_msgs::msg::TrackingError msg;
  const size_t n = std::min(msg.tracking_error.size(), data.size());
  std::copy_n(data.begin(), n, msg.tracking_error.begin());
  return msg;
}

/**
 * @brief Convert edge_msgs::msg::TrackingError to std::vector<double>.
 */
inline std::vector<double> TrackingErrorMsgToVector(const edge_msgs::msg::TrackingError& msg) {
  return std::vector<double>(msg.tracking_error.begin(), msg.tracking_error.end());
}

// ======================================================
// =============== Eigen ↔ ROS message ==================
// ======================================================

/**
 * @brief Convert Action message to Eigen 6D vector.
 */
inline Vector6d ActionMsgToEigen(const edge_msgs::msg::Action& msg) {
  return VectorToEigen<6>(ActionMsgToVector(msg));
}

/**
 * @brief Convert Eigen 6D vector to Action message.
 */
inline edge_msgs::msg::Action EigenToActionMsg(const Vector6d& v) {
  return VectorToActionMsg(EigenToVector(v));
}

/**
 * @brief Convert TrackingError message to Eigen 12D vector.
 */
inline Vector12d TrackingErrorMsgToEigen(const edge_msgs::msg::TrackingError& msg) {
  return VectorToEigen<12>(TrackingErrorMsgToVector(msg));
}

/**
 * @brief Convert Eigen 12D vector to TrackingError message.
 */
inline edge_msgs::msg::TrackingError EigenToTrackingErrorMsg(const Vector12d& v) {
  return VectorToTrackingErrorMsg(EigenToVector(v));
}

// ======================================================
// =============== Utility / Debug Helpers ==============
// ======================================================

/**
 * @brief Convert an Eigen matrix to a formatted string (for logging or debugging).
 */
template <typename Derived>
inline std::string EigenToString(const Eigen::MatrixBase<Derived>& mat, int precision = 4) {
  Eigen::IOFormat fmt(precision, 0, ", ", "\n", "[", "]");
  std::stringstream ss;
  ss << mat.format(fmt);
  return ss.str();
}

/**
 * @brief Safe pretty-print wrapper for ROS2 logging (thread-local buffer).
 * Example:
 * ```cpp
 * RCLCPP_INFO(logger, "Kp:\n%s", EigenToCString(Kp).c_str());
 * ```
 */
template <typename Derived>
inline const char* EigenToCString(const Eigen::MatrixBase<Derived>& mat, int precision = 4) {
  static thread_local std::string buffer;
  buffer = EigenToString(mat, precision);
  return buffer.c_str();
}

/**
 * @brief Convert Eigen matrix to YAML-style flat std::vector (for saving to config files).
 */
template <typename Derived>
inline std::vector<double> EigenToYAMLVector(const Eigen::MatrixBase<Derived>& mat) {
  return EigenToVector(mat);
}

}  // namespace edge::ros_interfaces
