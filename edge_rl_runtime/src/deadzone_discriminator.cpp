/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

// Implementation for DeadzoneDiscriminator
#include <iostream>
#include "edge_rl_runtime/deadzone_discriminator.hpp"

namespace edge::rl_runtime
{

DeadzoneDiscriminator::DeadzoneDiscriminator(double threshold, double y_left, double y_right, double x_front, double x_back)
    : threshold_(threshold), y_left_dist_(y_left), y_right_dist_(y_right), x_front_dist_(x_front), x_back_dist_(x_back)
{
    std::cout << "Deadzone initialized with threshold: " << threshold_
              << ", y_left_distance: " << y_left_dist_
              << ", y_right_distance: " << y_right_dist_
              << ", x_front_distance: " << x_front_dist_
              << ", x_back_distance: " << x_back_dist_
              << std::endl;
}

bool DeadzoneDiscriminator::is_in_deadzone(double pos_x, double pos_y) const
{
    // Placeholder implementation - replace with actual deadzone logic
    if (std::abs(pos_y) > y_left_dist_ || std::abs(pos_y) > y_right_dist_ ||
        std::abs(pos_x) > x_front_dist_ || std::abs(pos_x) > x_back_dist_)
    {
        return true;
    }
    return false;
}

} // namespace edge::rl_runtime