#pragma once
/**
 * @file conversions.hpp
 * @brief Helper functions for converting between Eigen vectors, std::vector, and edge_msgs ROS messages.
 *
 * This header provides lightweight inline conversion utilities
 * between Eigen (for control/learning), std::vector (for general containers),
 * and ROS message types (Action, TrackingError).
 */

#include <Eigen/Dense>
#include <vector>
#include "edge_msgs/msg/action.hpp"
#include "edge_msgs/msg/tracking_error.hpp"

namespace edge::ros_interfaces {

// -------------------- Type Aliases --------------------
using Vector6d  = Eigen::Matrix<double, 6, 1>;
using Vector12d = Eigen::Matrix<double, 12, 1>;

// ======================================================
// =============== Eigen ↔ std::vector ==================
// ======================================================

/**
 * @brief Convert Eigen vector (fixed-size or dynamic) to std::vector<double>.
 */
template <typename Derived>
inline std::vector<double> EigenToVector(const Eigen::MatrixBase<Derived>& v) {
  return std::vector<double>(v.derived().data(), v.derived().data() + v.size());
}

/**
 * @brief Convert std::vector<double> to Eigen::VectorXd.
 */
inline Eigen::VectorXd VectorToEigen(const std::vector<double>& vec) {
  Eigen::VectorXd v(vec.size());
  for (size_t i = 0; i < vec.size(); ++i)
    v(i) = vec[i];
  return v;
}

/**
 * @brief Convert std::vector<double> to fixed-size Eigen vector (compile-time dimension).
 */
template <int N>
inline Eigen::Matrix<double, N, 1> VectorToEigenFixed(const std::vector<double>& vec) {
  Eigen::Matrix<double, N, 1> v;
  for (int i = 0; i < N && i < static_cast<int>(vec.size()); ++i)
    v(i) = vec[i];
  return v;
}

// ======================================================
// =============== std::vector ↔ ROS msg ================
// ======================================================

/**
 * @brief Convert std::vector<double> to Action message.
 */
inline edge_msgs::msg::Action VectorToActionMsg(const std::vector<double>& data) {
  edge_msgs::msg::Action msg;
  for (size_t i = 0; i < msg.action.size() && i < data.size(); ++i) {
    msg.action[i] = data[i];
  }
  return msg;
}

/**
 * @brief Convert Action message to std::vector<double>.
 */
inline std::vector<double> ActionMsgToVector(const edge_msgs::msg::Action& msg) {
  std::vector<double> vec;
  for (size_t i = 0; i < msg.action.size(); ++i) {
    vec.push_back(msg.action[i]);
  }
  return vec;
}

/**
 * @brief Convert std::vector<double> to TrackingError message.
 */
inline edge_msgs::msg::TrackingError VectorToTrackingErrorMsg(const std::vector<double>& data) {
  edge_msgs::msg::TrackingError msg;
for (size_t i = 0; i < msg.tracking_error.size() && i < data.size(); ++i) {
    msg.tracking_error[i] = data[i];
}
  return msg;
}

/**
 * @brief Convert TrackingError message to std::vector<double>.
 */
inline std::vector<double> TrackingErrorMsgToVector(const edge_msgs::msg::TrackingError& msg) {
  std::vector<double> vec;
  for (size_t i = 0; i < msg.tracking_error.size(); ++i) {
    vec.push_back(msg.tracking_error[i]);
  }
  return vec;
}

// ======================================================
// =============== Eigen ↔ ROS msg ======================
// ======================================================

/**
 * @brief Convert Action message to Eigen 6D vector.
 */
inline Vector6d ActionMsgToEigen(const edge_msgs::msg::Action& msg) {
  return VectorToEigenFixed<6>(ActionMsgToVector(msg));
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
  return VectorToEigenFixed<12>(TrackingErrorMsgToVector(msg));
}

/**
 * @brief Convert Eigen 12D vector to TrackingError message.
 */
inline edge_msgs::msg::TrackingError EigenToTrackingErrorMsg(const Vector12d& v) {
  return VectorToTrackingErrorMsg(EigenToVector(v));
}

}  // namespace edge::ros_interfaces
