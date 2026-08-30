<div align="center">

# 🐾 Edge-Brain

### An On-device Runtime Learning Framework for Real-World Robots in Open-World Environments

[![Ubuntu 20.04/22.04](https://img.shields.io/badge/Ubuntu-20.04/22.04-red?logo=ubuntu)](https://ubuntu.com/)
[![ROS2 Foxy/Humble](https://img.shields.io/badge/ros2-foxy/humble-brightgreen.svg?logo=ros)](https://wiki.ros.org/foxy)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg?logo=apache)](https://opensource.org/license/apache-2-0)

[Paper](https://arxiv.org/abs/2503.04794) | [Docker Setup](#option-a-docker) | [Bare-Metal Setup](#option-b-bare-metal-native) | [Jetson Deployment](#jetson-deployment-guide) | [Acknowledgments](#acknowledgments)


</div>

Edge-Brain is a modular and scalable robotics framework designed for deploying learning-based control systems on real-world quadruped platforms.
It bridges simulation and real-world deployment, integrating learning, safety supervision, and system-level coordination into a unified architecture.

## Table of Contents

- [� Edge-Brain](#-edge-brain)
    - [An On-device Runtime Learning Framework for Real-World Robots in Open-World Environments](#an-on-device-runtime-learning-framework-for-real-world-robots-in-open-world-environments)
  - [Table of Contents](#table-of-contents)
  - [💡 User Guide](#-user-guide)
    - [⚙️ Dependencies](#️-dependencies)
    - [🔨 Setup](#-setup)
      - [🐳 Option A: Docker](#-option-a-docker)
        - [A.1 Build it yourself](#a1-build-it-yourself)
        - [A.2 Pull the pre-built image](#a2-pull-the-pre-built-image)
      - [📟 Option B: Bare-Metal (Native)](#-option-b-bare-metal-native)
    - [🔧 Build Script](#-build-script)
  - [🏷️ Misc](#️-misc)
  - [🧩 Features](#-features)
  - [💻 Jetson Deployment Guide](#-jetson-deployment-guide)
  - [🎉 Acknowledgments](#-acknowledgments)

---

## 💡 <a id="user-guide"></a>User Guide

### ⚙️ <a id="dependencies"></a>Dependencies

* ROS2 Humble
* Jetpack 6.2
* L4T (Linux for Tegra) r36.4.3

---

Package Dependency:

* *Coal (Extension of the Flexible Collision Library) - 3.0.1*
* *pinocchio (Rigid Body Dynamics Library) - 3.7.0*
* *Mujoco - 3.2.7*


### 🔨 <a id="setup"></a>Setup

1. Clone this repository:

```bash
mkdir -p ~/edge_ws/src && cd ~/edge_ws/src
git clone -b Jetson https://github.com/Charlescai123/edge-brain-go2.git
```

2. Initialize and update all submodules recursively
```bash
cd edge-brain-go2
git submodule update --init --recursive
```

3. Install, using **one** of the two options below.

Two installation paths are supported: **Docker** (recommended — reproducible, isolated) and **Bare-Metal** (native install on the host, useful for development or when Docker overhead isn't wanted).

#### 🐳 <a id="option-a-docker"></a>Option A: Docker

Two ways to get the Docker image, using **one** of the two options below.

##### A.1 Build it yourself

The `docker/` folder auto-detects your platform (`x86_64` / `arm64` / Jetson) and builds a base image (ROS2 + CUDA + robotics libs, see `docker/dockerfile/base_*.Dockerfile`) followed by an app image (Gazebo, extra ROS2 packages, Python deps, and this workspace, see `docker/dockerfile/app.Dockerfile`).

```bash
cd docker/scripts

# Builds edge-brain-base:<platform> then edge-brain-go2:<platform>
./build.sh

# Run the container (auto-selects --gpus/--runtime nvidia per platform)
./start.sh
```

**Note:** `build.sh` clones this repo again *inside* the app image over SSH (`git@github.com:...`), so make sure a GitHub-registered SSH key exists at `~/.ssh/id_rsa` before building — the script starts `ssh-agent` and forwards it into the build automatically.

##### A.2 Pull the pre-built image

<!-- TODO: publish edge-brain-go2 images and fill in the registry path/tags below -->

```bash
# TODO: replace with the published image, e.g. ghcr.io/<org>/edge-brain-go2:<platform>
docker pull <TODO-registry>/edge-brain-go2:<platform>

# start.sh expects the local tag edge-brain-go2:<platform>, so re-tag the pulled image
docker tag <TODO-registry>/edge-brain-go2:<platform> edge-brain-go2:<platform>

cd docker/scripts
./start.sh
```

#### 📟 <a id="option-b-bare-metal-native"></a>Option B: Bare-Metal (Native)

**Prerequisites:** Ubuntu 22.04, [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html), and either the CUDA 12.6 Toolkit (x86_64/dGPU) or a flashed JetPack 6.2 / L4T r36.4.3 (Jetson).

<details>
<summary>Click to expand full native install steps (mirrors <code>docker/dockerfile/base_x86_64.Dockerfile</code> + <code>app.Dockerfile</code>)</summary>

```bash
# 1) System + ROS2 build tools
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    curl wget gnupg2 lsb-release ca-certificates git vim build-essential \
    cmake python3 python3-pip locales iproute2 net-tools bash-completion \
    software-properties-common nmap lsof libglfw3-dev locate \
    liburdfdom-headers-dev liburdfdom-dev liboctomap-dev

sudo apt-get install -y --no-install-recommends \
    python3-rosdep python3-rosinstall python3-vcstool python3-colcon-common-extensions \
    ros-humble-desktop ros-humble-backward-ros ros-humble-common-interfaces \
    ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-octomap \
    ros-humble-octomap-msgs ros-humble-rmw-fastrtps-cpp ros-humble-rmw-cyclonedds-cpp

sudo rosdep init || true
rosdep update

# 2) Third-party C++ libraries
mkdir -p ~/third_party && cd ~/third_party

git clone https://github.com/stack-of-tasks/eigenpy.git
cd eigenpy && mkdir build && cd build && cmake .. && make -j$(nproc) && sudo make install
cd ~/third_party

sudo apt install -y libeigen3-dev libboost-all-dev python3-dev
git clone --recursive https://github.com/coal-library/coal.git
cd coal && git checkout devel && git submodule update --init --recursive
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && sudo make install
cd ~/third_party

git clone --recursive https://github.com/humanoid-path-planner/hpp-fcl.git
cd hpp-fcl && git checkout v2.4.5 && git submodule update --init --recursive
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && sudo make install
cd ~/third_party

git clone --recursive https://github.com/stack-of-tasks/pinocchio.git
cd pinocchio && git submodule update --init --recursive
mkdir build && cd build && cmake .. \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_INTERFACE=ON \
    -DBUILD_WITH_COLLISION_SUPPORT=ON -DPYTHON_EXECUTABLE=$(which python3) \
    -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc) && sudo make install
cd ~/third_party

git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2 && git checkout f29ee9f
mkdir build && cd build && cmake .. && make -j$(nproc) && sudo make install
cd ~/third_party

sudo apt install -y ros-humble-rmw-cyclonedds-cpp ros-humble-rosidl-generator-dds-idl libyaml-cpp-dev
git clone https://github.com/unitreerobotics/unitree_ros2.git
cd unitree_ros2/cyclonedds_ws/src
git clone https://github.com/ros2/rmw_cyclonedds -b humble
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd .. && colcon build --packages-select cyclonedds
source /opt/ros/humble/setup.bash && colcon build
cd ~/third_party

git clone https://github.com/google-deepmind/mujoco.git
cd mujoco && git checkout 3.3.3
mkdir build && cd build && cmake .. && make -j$(nproc) && sudo make install
cd ~/third_party

git clone https://github.com/Charlescai123/unitree_mujoco.git
cd unitree_mujoco/simulate && git checkout mujoco-v3.3.3
mkdir build && cd build && cmake .. && make -j$(nproc)

# 3) PyTorch + environment variables
pip3 install --upgrade pip
pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH' >> ~/.bashrc
source ~/.bashrc

# 4) Gazebo (Ignition Fortress) + extra ROS2 packages
sudo apt-get install -y lsb-release gnupg
curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/gazebo-stable.list
sudo apt-get update && sudo apt-get install -y ignition-fortress

sudo apt update && sudo apt install -y ros-humble-pcl-ros ros-humble-pcl-conversions \
    ros-humble-xacro ros-humble-ros-gz-bridge ros-humble-joint-state-publisher-gui \
    ros-humble-ros-gz-sim ros-humble-ros-gz-interfaces ros-humble-rqt-robot-steering

# 5) Python dependencies
pip3 install --ignore-installed transforms3d ultralytics cv-bridge opencv-python \
    numpy==1.26.4 segment-anything pdaqp==0.6.7 cvxpy==1.7.3 cvxopt==1.3.2 scipy==1.15.3 setuptools==59.6.0

# 6) Gain synthesizer solver (cvxpygen)
cd ~/edge_ws/src/edge-brain-go2/edge_locomotion/edge_phy_teacher/src/optimization/ecvxcone/third_party/cvxpygen
pip install --user -e . && python build_solver.py

# 7) Build the ROS 2 workspace (see Build Script section below)
source /opt/ros/humble/setup.bash
cd ~/edge_ws/src/edge-brain-go2
./edge_scripts/build_ws.sh -w ~/edge_ws
source ~/edge_ws/install/setup.bash
```

For ARM64 or Jetson bare-metal installs, mirror the equivalent steps in `docker/dockerfile/base_arm64.Dockerfile` / `docker/dockerfile/base_jetson.Dockerfile` instead of `base_x86_64.Dockerfile`, and see the [Jetson Deployment Guide](#jetson-deployment-guide) below for JetPack/L4T/RT-kernel setup.

</details>

### 🔧 <a id="build-script"></a>Build Script

`edge_scripts/build_ws.sh` compiles the ROS 2 workspace's custom packages (used by both install paths — internally by the Docker app image, and directly for bare-metal):

```bash
cd ~/edge_ws/src/edge-brain-go2

# Auto-detect workspace and build
./edge_scripts/build_ws.sh

# Specify workspace manually
./edge_scripts/build_ws.sh -w ~/edge_ws

# Show help
./edge_scripts/build_ws.sh -h
```

**Options:**
- `-w, --workspace DIR` : Specify the ROS 2 workspace directory (must contain `edge_ws` in name)
- `-h, --help` : Display help message

**Note:** Ensure the script has execute permissions: `chmod +x edge_scripts/build_ws.sh`


---

## 🏷️ <a id="misc"></a>Misc

---

Digital Twin Video (In Rviz):

https://github.com/user-attachments/assets/6d2f6dc1-04ef-45a7-abfa-cc55f1900507

## 🧩 <a id="features"></a>Features

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

## 💻 <a id="jetson-deployment-guide"></a>Jetson Deployment Guide

To ensure optimal performance on Jetson platforms, it is strongly recommended to use an NVMe SSD instead of the onboard 64GB eMMC. NVMe storage provides significantly higher read/write throughput. After installing the NVMe drive, use [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) to flash the JetPack SDK directly onto the NVMe device.

In addition, the following configurations are recommended for stable and high-performance deployment:
* [Build the Real-Time Kernel](./docs/rt-kernel/)
* [Setup AP Mode (Auto Boot)](./docs/jetson-cfg/ap-mode/)
* [Enable Maximum Performance Mode](./docs/jetson-cfg/max-perf/)
* [Configure Remote Display Port](./docs/jetson-cfg/remote-dp/)

---

## 🎉 <a id="acknowledgments"></a>Acknowledgments

Special thanks to the contributions from these repos:

- [quadruped_ros2_control](https://github.com/legubiao/quadruped_ros2_control): ROS2 Framework for ocs-based mpc control.