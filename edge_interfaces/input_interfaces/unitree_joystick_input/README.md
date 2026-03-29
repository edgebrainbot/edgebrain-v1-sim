# Unitree Joystick Input Node

This node will listen to the wireless remote topic and publish a `unitree_go::msg::dds_::WirelessController_` message by using `unitree_sdk2`.

> Before use this node, please use `ifconfig` command to check the network interface to connect to the robot, then change the parameter in launch file.

Tested environment:
* Ubuntu 24.04
  * ROS2 Jazzy

### Build Command
```bash
cd ~/ros2_ws
colcon build --packages-up-to unitree_joystick_input --symlink-install
```

### Launch Command
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch unitree_joystick_input joystick.launch.py
```

Launch arguments:
* `network_interface` (default: empty string): network interface for Unitree DDS. Leave empty to auto-select.
* `domain_id` (default: `0`): DDS domain ID used by Unitree DDS runtime.

Examples:
```bash
# Auto-select interface
ros2 launch unitree_joystick_input joystick.launch.py network_interface:="" domain_id:=0

# Force interface explicitly
ros2 launch unitree_joystick_input joystick.launch.py network_interface:=eno1 domain_id:=0
```

## 1. Use Instructions for Controllers

### 1.1 Unitree Guide Controller
* Passive Mode: select
* Fixed Down: start
* Fixed Stand: start
    * Free Stand: right + X
    * Trot: right + Y
    * SwingTest: right + B
    * Balance: right + A

### 1.2 OCS2 Quadruped Controller
* Passive Mode: select
* OCS2 Mode: start
  * Stance: start 
  * trot: right + X
  * standing trot: right + Y
  * flying_trot: right + B

### 1.3 RL Quadruped Controller
* Passive Mode: select
* Fixed Down: start
* Fixed Stand: start
  * RL Mode: right + X