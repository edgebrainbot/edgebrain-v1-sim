#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "edge_dummy_env/go2_high_state_subscriber.hpp"

int main(int argc, char** argv) {
    const std::string network_interface = (argc > 1) ? argv[1] : "eno1";

    edge::dummy_env::Go2HighStateSubscriber subscriber(network_interface);
    subscriber.Init();

    std::cout << "[go2_high_state_sub_test] listening on nic=" << network_interface << std::endl;

    while (true) {
        subscriber.PrintLatestState();
        // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    return 0;
}
