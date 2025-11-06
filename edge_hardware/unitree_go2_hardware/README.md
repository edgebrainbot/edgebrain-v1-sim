# Unitree Go2 Hardware

A ros2-control-based package providing the hardware interface for the Go2 quadruped robot, compatible with both the Mujoco simulator and real Unitree robots via unitree_sdk2.

Please use mujoco simulation in [unitree_mujoco](https://github.com/legubiao/unitree_mujoco). In this simulation, I add foot force sensor support.

### ⚙️ Dependencies

* [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)

## 1. Interfaces

Required hardware interfaces:

* command:
  * joint position
  * joint velocity
  * joint effort
  * KP
  * KD
* state:
  * joint effort
  * joint position
  * joint velocity
  * imu sensor
    * linear acceleration
    * angular velocity
    * orientation
    * rpy
    * temperature
  * foot force sensor
  * odometer
    * position 
    * linear velocity

## 2. Build

Build Command:
```bash
cd ~/ros2_ws
colcon build --packages-up-to unitree_go2_hardware --symlink-install
```

## 3. Config network and domain
The real unitree robot has different network and domain name, you need to set the network and domain name in the [xacro](../../edge_description/unitree/go2_description/xacro/go2.ros2_control.xacro) file each time you run the hardware.
```xml
<hardware>
    <plugin>unitree_go2_hardware/Go2HardwareInterface</plugin>
    <!-- <param name="domain">0</param> 
    <param name="network_interface">eno1</param>
    <param name="show_foot_force">true</param> -->
</hardware>
```

After modified the config, you can tried to visualize the robot info from real robot by following command:
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch unitree_go2_hardware visualize.launch.py
```


## Misc

Added efficient implementations of the original CRC32 algorithm. To benchmark and compare the performance differences, run:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run unitree_go2_hardware test_crc32