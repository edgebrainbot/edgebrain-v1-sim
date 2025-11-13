#include <cmath>
#include <iostream>
#include "edge_config/reward_composer.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

RewardComposer::RewardComposer(const std::string& yaml_path)
{
    std::string final_path;

    if (yaml_path.empty()) {
        std::string pkg_share = ament_index_cpp::get_package_share_directory("edge_config");
        final_path = pkg_share + "/config/env_reward.yaml";

        std::cout << "[RewardComposer] Using default reward file: " 
                  << final_path << "\n";
    } else {
        final_path = yaml_path;
        std::cout << "[RewardComposer] Using custom reward file: "
                  << final_path << "\n";
    }
    YAML::Node node = YAML::LoadFile(final_path);

    if (node["reward"]) {
        for (auto it : node["reward"]) {
            std::string key = it.first.as<std::string>();
            double s = it.second.as<double>();
            scales_[key] = s;
        }
    }

    // init P matrix (replace with your MATRIX_P)
    P_ = Eigen::Map<const Eigen::Matrix<double, 10, 10, Eigen::RowMajor>>(&MATRIX_P[0][0]);

    std::cout << "[RewardComposer] Loaded reward scales:\n";
    for (auto& kv : scales_) {
        std::cout << "  - " << kv.first << ": " << kv.second << "\n";
    }
}

double RewardComposer::computeTotalReward(
    const Eigen::VectorXd& err_prev,
    const Eigen::VectorXd& err_next
    // int is_fail,
    // const Eigen::VectorXd& delta_action,
    // double forward_speed,
    // double body_contact,
    // double energy,
    // double grav_z
) {
    double total = 0.0;

    for (auto& kv : scales_) {
        const std::string& key = kv.first;
        double scale = kv.second;

        double r = 0.0;

        // if (key == "fall_down") r = fall_down(is_fail);
        if (key == "lin_vel_tracking") r = lin_vel_tracking(err_prev);
        else if (key == "ang_vel_tracking") r = ang_vel_tracking(err_prev);
        else if (key == "orientation_tracking") r = orientation_tracking(err_prev);
        else if (key == "height_tracking") r = height_tracking(err_prev);
        // else if (key == "jerky_action") r = jerky_action(delta_action);
        // else if (key == "forward_speed") r = forward_speed_rew(forward_speed);
        // else if (key == "body_contact") r = body_contact_rew(body_contact);
        // else if (key == "energy_consumption") r = energy_consumption(energy);
        // else if (key == "upright") r = upright(grav_z);
        else if (key == "lyapunov") r = lyapunov(err_prev, err_next);

        // else {
        //     std::cout << "[RewardComposer] Unknown reward key: " << key << "\n";
        //     continue;
        // }

        total += r * scale;
    }

    std::cout << "[RewardComposer] Total reward: " << total << "\n";
    return total;
}


double RewardComposer::fall_down(int is_fail) {
    return -static_cast<double>(is_fail);
}

double RewardComposer::lin_vel_tracking(const Eigen::VectorXd& err) {
    double e = err(6)*err(6) + err(7)*err(7) + err(8)*err(8);
    return std::exp(-e / 0.2);
}

double RewardComposer::ang_vel_tracking(const Eigen::VectorXd& err) {
    double e = err(9)*err(9) + err(10)*err(10) + err(11)*err(11);
    return std::exp(-e / 0.2);
}

double RewardComposer::orientation_tracking(const Eigen::VectorXd& err) {
    return -(err(3)*err(3) + err(4)*err(4) + err(5)*err(5));
}

double RewardComposer::height_tracking(const Eigen::VectorXd& err) {
    return -std::abs(err(2));
}

double RewardComposer::jerky_action(const Eigen::VectorXd& delta_action) {
    return delta_action.sum();
}

double RewardComposer::forward_speed_rew(double speed) {
    return speed;
}

double RewardComposer::body_contact_rew(double c) {
    return -c;
}

double RewardComposer::energy_consumption(double en) {
    return -en;
}

double RewardComposer::upright(double gz) {
    return gz;
}

double RewardComposer::lyapunov(
    const Eigen::VectorXd& err_prev,
    const Eigen::VectorXd& err_next
) {
    Eigen::VectorXd sp = err_prev.tail(10);
    Eigen::VectorXd sn = err_next.tail(10);

    double v1 = sp.transpose() * P_ * sp;
    double v2 = sn.transpose() * P_ * sn;

    return v1 - v2;
}
