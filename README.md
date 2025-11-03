# Edge-Learning-Go2

[![Ubuntu 20.04/22.04](https://img.shields.io/badge/Ubuntu-20.04/22.04-red?logo=ubuntu)](https://ubuntu.com/)
[![ROS2 Foxy/Humble](https://img.shields.io/badge/ros2-foxy/humble-brightgreen.svg?logo=ros)](https://wiki.ros.org/foxy)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg?logo=apache)](https://opensource.org/license/apache-2-0)

---

🔧

## 💡 User Guide

### ⚙️ Dependencies

* ROS2 Humble
* Jetpack 6.2
* L4T (Linux for Tegra) r36.4.3

---

We recommend purchasing an NVMe SSD, as it offers significantly faster read and write speeds compared to the built-in 64GB eMMC. Once installed, use the [SDK Manager](https://developer.nvidia.com/sdk-manager) to flash the JetPack SDK directly onto the NVMe drive.


> **Real-time (RT) Kernel:** To patch the RT kernel, please refer to [Jetson-RT-Kernel](https://github.com/Charlescai123/Jetson-RT-Kernel) for more details.


For remote onboard Jetson debug, you can use other remote connection tool (e.g., *Anydesk*, *Teamviewer*). However, these softwares are not typically useful if the Jetson is not connected to a monitor. According to the [solution](https://askubuntu.com/questions/453109/add-fake-display-when-no-monitor-is-plugged-in), there're two options:

1. purchase a dummy display plug [DP Port](https://www.amazon.com/FUERAN-DP-DisplayPort-emulator-2560x1600/dp/B075PTQ4NH/ref=sr_1_2_sspa?dib=eyJ2IjoiMSJ9._2VBeAusIsfeL9L3oHOyN6PCekAraiSOV3Mbqo9epsJwtAFnpeQKXHkz0wJTW8nPMM9W9X-Z_Sbt1gaplwMw2RqV9BnFry5G4bcKFI4PVCb8FD_8RxNh8B5D97RcYCU7aVqnbDOkPJZwz5UVYRL-jNVwVQtxFcuWtiiWVWm0XvcXtdtGp9HA9WySLN1MFOEmpgGPhkNvBDvJuxqC3YC28CEOz7CkhmC0gsXiwtJNmkA.K023GTNCJhfrzl89n0sznwDJ5qqfZ3KOK0XIoN7h2XE&dib_tag=se&keywords=dp%2Bport%2Bdummy%2Bport&qid=1750013547&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1)
2. turn the video output to a 
```bash
sudo apt-get install xserver-xorg-video-dummy
```

Then add the content in `/etc/X11/xorg.conf`:
```bash
Section "Device"
    Identifier  "Configured Video Device"
    Driver      "dummy"
    # Default is 4MiB, this sets it to 16MiB
    VideoRam    16384
EndSection

Section "Monitor"
    Identifier  "Configured Monitor"
    HorizSync 31.5-48.5
    VertRefresh 50-70
EndSection

Section "Screen"
    Identifier  "Default Screen"
    Monitor     "Configured Monitor"
    Device      "Configured Video Device"
    DefaultDepth 24
    SubSection "Display"
    Depth 24
    Modes "1024x800"
    EndSubSection
EndSection
```

Package Dependency:

* *Coal (Extension of the Flexible Collision Library) - 3.0.1*
* *pinocchio (Rigid Body Dynamics Library) - 3.7.0*
* *Mujoco - 3.2.7*


### 🔨 Setup

1. Clone this repository:

```bash
git clone -b humble git@github.com:Charlescai123/edge-learning-go2.git
```

2. Initialize and update all submodules recursively
```bash
git submodule  update --init --recursive
```

3. Use docker to build the image:
```bash
docker build -t edge-brain:humble -f ./docker/Dockerfile .
```

## ⏳ To Do ##

<!-- * [x] Add BEV map to the repo -->
<!-- * [x] Fast Marching Method (FMM) implementation -->

* [ ] Code Refactorization
* [ ] Incorporate more challenging scenarios
    * [x] Dense forests (sandy terrain, trees)
    * [ ] inclined staircases, and rainy conditions
* [ ] Go2 real robot deployment
    * [ ] Gazebo real-time testing
    * [ ] ROS/ROS2 integration
* [ ] Restructure the code as FSM and add teleoperation (optional)
* [ ] Migration to Isaac-Lab

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

## 📝 Citation

Please star or cite below papers if you find this repo helpful 🙏

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
```

---

## 🎉 Acknowledgments

Special thanks to the contributions from these repos:

- [quadruped_ros2_control](https://github.com/legubiao/quadruped_ros2_control): ROS2 Framework for ocs-based mpc control.
