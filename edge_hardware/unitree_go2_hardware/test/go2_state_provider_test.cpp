#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "unitree_go2_hardware/api/providers/go2_state_provider.hpp"

int main(int argc, char** argv)
{
    const std::string network_interface =
        (argc > 1) ? argv[1] : "eno1";

    edge::go2_api::provider::Go2StateProvider state_provider(network_interface);

    state_provider.Init();

    std::cout << "[go2_state_provider_test] listening on nic="
              << network_interface << std::endl;

    while (true)
    {
        if (state_provider.HasState())
        {
            auto state = state_provider.GetLatestVec12Eigen();

            std::cout << "position: "
                      << state(0) << " "
                      << state(1) << " "
                      << state(2) << std::endl;

            std::cout << "rpy: "
                      << state(3) << " "
                      << state(4) << " "
                      << state(5) << std::endl;

            std::cout << "velocity: "
                      << state(6) << " "
                      << state(7) << " "
                      << state(8) << std::endl;

            std::cout << "gyro: "
                      << state(9) << " "
                      << state(10) << " "
                      << state(11) << std::endl;

            std::cout << "-------------------------" << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    return 0;
}