# ======================================================
# Edge-Brain ROS2 Base Image for Jetson / x86 GPU
# ======================================================

# ---------- Arguments ----------
ARG CUDA_VERSION=12.6.0
ARG UBUNTU_VERSION=22.04

# ---------- Base Image ----------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

LABEL maintainer="Yihao Cai <yihao.cai@wayne.edu>" \
      version="v2.0.0" \
      description="ROS2 Humble + CUDA 12.6 + Robotics Libraries" \
      license="Apache-2.0"


# ---------- System Environment ----------
ARG USERNAME=edgebot
ARG USER_UID=1000

ENV ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8


# ---------- Basic Tools -----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget gnupg2 lsb-release ca-certificates git vim build-essential \
    cmake python3 python3-pip locales iproute2 net-tools bash-completion \
    software-properties-common nmap lsof libglfw3-dev locate liburdfdom-headers-dev \
    liburdfdom-dev liboctomap-dev \
    && locale-gen en_US.UTF-8


# ---------- Install ROS2 -----------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-desktop ros-${ROS_DISTRO}-backward-ros ros-${ROS_DISTRO}-common-interfaces \
    ros-${ROS_DISTRO}-ros2-control ros-${ROS_DISTRO}-ros2-controllers ros-humble-octomap \
    ros-humble-octomap-msgs \
    python3-rosdep python3-rosinstall python3-vcstool python3-colcon-common-extensions

RUN rosdep init || true && rosdep update


# ---------- Create user -----------
RUN useradd -m ${USERNAME} && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ${USERNAME}
WORKDIR /home/${USERNAME}


# ----------- Copy ROS2 Workspace ------------
ARG ROS2_WS_NAME=edge_ws
ARG ROS2_WS_PATH=/home/${USERNAME}/${ROS2_WS_NAME}
COPY --chown=${USERNAME}:${USERNAME} ../ ${ROS2_WS_PATH}/src/
SHELL ["/bin/bash", "-c"]


# ----------- Build unitree_sdk2 -------------
RUN cd ${ROS2_WS_PATH}/src/third_party/unitree_sdk2 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    sudo make install


# ----------- Build unitree_ros2 -------------
RUN sudo apt update && \
    sudo apt install -y ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-rosidl-generator-dds-idl libyaml-cpp-dev && \
    cd ${ROS2_WS_PATH}/src/third_party/unitree_ros2/cyclonedds_ws/src && \
    git clone https://github.com/ros2/rmw_cyclonedds -b ${ROS_DISTRO} && \
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x && \
    cd .. && colcon build --packages-select cyclonedds && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build


# ----------- Build mujoco -------------
RUN cd ${ROS2_WS_PATH}/src/third_party/mujoco && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    sudo make install


# ----------- Build unitree_mujoco -------------
RUN cd ${ROS2_WS_PATH}/src/third_party/unitree_mujoco/simulate && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)


# ----------- Build eigenpy -------------
RUN git clone https://github.com/stack-of-tasks/eigenpy.git && \
    cd eigenpy && mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    sudo make install


# ----------- Build Coal -------------
RUN sudo apt update && \
    sudo apt install -y cmake libeigen3-dev libboost-all-dev python3-dev python3-pip && \
    git clone --recursive https://github.com/coal-library/coal.git && \
    cd coal && git checkout devel && \
    git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    sudo make install


# ---------- Build hpp-fcl -------------
RUN git clone --recursive https://github.com/humanoid-path-planner/hpp-fcl.git && \
    cd hpp-fcl && git checkout v2.4.5 && \
    git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    sudo make install


# ----------- Build pinocchio -------------
RUN git clone --recursive https://github.com/stack-of-tasks/pinocchio.git && \
    cd pinocchio && git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_INTERFACE=ON \
    -DBUILD_WITH_COLLISION_SUPPORT=ON \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    sudo make install


# ----------- Set Paths -------------
RUN echo '########## Add Pinocchio Library ##########' >> .bashrc
RUN echo 'export PATH=/usr/local/bin:$PATH' >> .bashrc
RUN echo 'export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH' >> .bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> .bashrc
RUN echo 'export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH' >> .bashrc
RUN echo 'export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH' >> .bashrc
RUN echo '###########################################' >> .bashrc


# ----------- Set Environment -------------
ENV ROS_DISTRO=${ROS_DISTRO}
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PYTHONPATH=/usr/lib/python3/dist-packages:/home/edgebot/ros2_ws/install/lib/python3.10/site-packages


# ----------- Install PyTorch GPU for JetPack 6.2 -------------
# reference: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126


# ----------- Build ROS2 Workspace -------------
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd ${ROS2_WS_PATH} && colcon build --packages-up-to edge_quadruped_controller --symlink-install || true


# ----------- Entrypoint -------------
COPY ./docker/entrypoint.sh /entrypoint.sh
# RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
CMD ["bash"]
