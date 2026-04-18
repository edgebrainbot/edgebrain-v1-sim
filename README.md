# Edge-Brain

[![Ubuntu 20.04/22.04](https://img.shields.io/badge/Ubuntu-20.04/22.04-red?logo=ubuntu)](https://ubuntu.com/)
[![ROS2 Foxy/Humble](https://img.shields.io/badge/ros2-foxy/humble-brightgreen.svg?logo=ros)](https://wiki.ros.org/foxy)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg?logo=apache)](https://opensource.org/license/apache-2-0)

Edge-Brain is a modular and scalable robotics framework designed for deploying learning-based control systems on real-world quadruped platforms.
It bridges simulation and real-world deployment, integrating learning, safety supervision, and system-level coordination into a unified architecture.

## 💡 User Guide

### ⚙️ Dependencies

* ROS2 Humble
* Jetpack 6.2
* L4T (Linux for Tegra) r36.4.3

---

Package Dependency:

* *Coal (Extension of the Flexible Collision Library) - 3.0.1*
* *pinocchio (Rigid Body Dynamics Library) - 3.7.0*
* *Mujoco - 3.2.7*


### 🔨 Setup

1. Clone this repository:

```bash
mkdir -p ~/edge_ws/src && cd ~/edge_ws/src
git clone -b Jetson https://github.com/Charlescai123/edge-brain-go2.git
```

2. Initialize and update all submodules recursively
```bash
git submodule  update --init --recursive
```

3. Use docker to build the image:
```bash
docker build -t edge-brain:humble -f ./docker/Dockerfile .
```

### 🔧 Build Script

A convenient build script is provided to compile the ROS 2 workspace:

```bash
# Auto-detect workspace and build
./scripts/build.sh

# Specify workspace manually
./scripts/build.sh -w ~/edge_ws

# Show help
./scripts/build.sh -h
```

**Options:**
- `-w, --workspace DIR` : Specify the ROS 2 workspace directory (must contain `ros2_ws` in name)
- `-h, --help` : Display help message

**Note:** Ensure the script has execute permissions: `chmod +x scripts/build.sh`


---

## 🏷️ Misc

---

Digital Twin Video (In Rviz):

https://github.com/user-attachments/assets/6d2f6dc1-04ef-45a7-abfa-cc55f1900507

## Features

There're several features that this repository shares:

- Modularity:
The system is organized into well-defined modules, each responsible for a specific functionality such as data collection, policy learning, safety supervision, and communication. This modular design enables independent development, testing, and replacement of components.

- Scalability:
The architecture supports scaling across different robot platforms and computational setups, from simulation to real-world deployment on embedded devices. Parallelized data pipelines and asynchronous execution allow efficient scaling with increasing computational resources.

- Interoperability:
The system interfaces seamlessly with ROS 2, PyTorch, and simulation frameworks (e.g., Isaac Gym / Isaac Sim), facilitating smooth integration with both existing robotics stacks and custom pipelines.

<!-- ## 📝 Citation

Please star or cite below papers if you find this repo helpful

```
@misc{cai2025runtimelearningquadrupedrobots,
      title={Runtime Learning of Quadruped Robots in Wild Environments}, 
      author={Yihao Cai and Yanbing Mao and Lui Sha and Hongpeng Cao and Marco Caccamo},
      year={2025},
      eprint={2503.04794},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2503.04794}, 
}
```

```
@misc{cao2024simplexenabledsafecontinuallearning,
      title={Simplex-enabled Safe Continual Learning Machine}, 
      author={Hongpeng Cao and Yanbing Mao and Yihao Cai and Lui Sha and Marco Caccamo},
      year={2024},
      eprint={2409.05898},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.05898}, 
}
``` -->

## 🖥️ Jetson Deployment Guide

To ensure optimal performance on Jetson platforms, it is strongly recommended to use an NVMe SSD instead of the onboard 64GB eMMC. NVMe storage provides significantly higher read/write throughput. After installing the NVMe drive, use [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) to flash the JetPack SDK directly onto the NVMe device.

In addition, the following configurations are recommended for stable and high-performance deployment:
* [Build the Real-Time Kernel](./docs/rt-kernel/)
* [Setup AP Mode (Auto Boot)](./docs/jetson-cfg/ap-mode/)
* [Enable Maximum Performance Mode](./docs/jetson-cfg/max-perf/)
* [Configure Remote Display Port](./docs/jetson-cfg/remote-dp/)

---

## 🎉 Acknowledgments

Special thanks to the contributions from these repos:

- [quadruped_ros2_control](https://github.com/legubiao/quadruped_ros2_control): ROS2 Framework for ocs-based mpc control.