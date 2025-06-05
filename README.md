# Edge-Learning-Go2

[![Ubuntu 20.04/22.04](https://img.shields.io/badge/Ubuntu-20.04/22.04-red?logo=ubuntu)](https://ubuntu.com/)
[![ROS2 Foxy/Humble](https://img.shields.io/badge/ros2-foxy/humble-brightgreen.svg?logo=ros)](https://wiki.ros.org/foxy)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg?logo=apache)](https://opensource.org/license/apache-2-0)

---


## 💡 User Guide

### ⚙️ Dependencies

* ROS2 Humble
* Jetpack 6.2
* L4T (Linux for Tegra) r36.4.3

---

We recommend purchasing an NVMe SSD, as it offers significantly faster read and write speeds compared to the built-in 64GB eMMC. Once installed, use the [SDK Manager](https://developer.nvidia.com/sdk-manager) to flash the JetPack SDK directly onto the NVMe drive.


Real-Time Kerenel Patch:

> **Note:** The real-time (RT) kernel enables in-response CPU scheduling, resulting in improved system responsiveness for real-time applications. We strongly recommend that users apply the RT kernel patch for their system’s real-time performance.


> **Warning:** Nvidia has provided the rt-kernel with a Debian package management-based [OTA](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Kernel/KernelCustomization.html#real-time-kernel-using-ota-update). But since it has been reported to cause the system to [freeze at boot](https://forums.developer.nvidia.com/t/boot-freezing-when-installing-preemptrt-on-nvme-setup-with-agx-orin-dev-kit-jetpack-6-2/323869) when entering the GUI, we recommend building and installing the kernel manually.

* Follow the [tutorials](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Kernel/KernelCustomization.html#sd-kernel-kernelcustomization) for compiling the kernel. The entire procedure typically generate:
    * A bindary real-time kernel image
    * The kernel module 5.15.148-rt-tegra
    * A set of DTB (Device Tree Blob) files for boot configuration

* (Optional) If UEFI Secure Boot is not enabled, you can skip the kernel signing and encryption steps. Or otherwise, follow [Secure Boot](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Security/SecureBoot.html#sd-security-secureboot).

* To install the kernel, there're two options:
    * Flash the new kernel and modules onto the Jetson device using `flash.sh`

    * Copy the kernel image, modules, and DTB files manually to the Jetson root filesystem


Here are some useful links for reference: 
- https://chipnbits.github.io/content/projects/RLUnicycle/rtkernel/rtpatch.html
- https://forums.developer.nvidia.com/t/preempt-rt-patches-for-jetson-nano/72941
- https://forums.developer.nvidia.com/t/jetson-agx-orin-rt-linux-without-reflashing/283832
- https://forums.developer.nvidia.com/t/no-display-with-preempt-rt-patches/240876
- https://forums.developer.nvidia.com/t/build-the-real-time-kernel/229571
- https://blog.csdn.net/weixin_43854380/article/details/126584835
- https://github.com/kozyilmaz/nvidia-jetson-rt/blob/master/docs/README.03-realtime.md


### 🔨 Setup

1. Clone this repository:

```bash
git clone git@github.com:Charlescai123/edge-learning-go2.git
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
