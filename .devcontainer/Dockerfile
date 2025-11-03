# ======================================================
# Edge-Brain ROS2 Base Image for Jetson / x86 GPU
# ======================================================

# ---------- Arguments ----------
ARG CUDA_VERSION=12.6.0
ARG UBUNTU_VERSION=22.04

# ---------- Base Image ----------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# ---------- System Environment ----------
ARG USERNAME=edgebot
ARG USER_UID=1000

ENV ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ---------- [1] Basic Tools ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget gnupg2 lsb-release ca-certificates git vim \
    build-essential cmake python3 python3-pip locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ---------- [2] Add ROS2 Source ----------
# (Use keyring instead of deprecated apt-key)
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# ---------- [3] Install ROS2 + Build Tools ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-desktop-full \
    python3-rosdep python3-rosinstall python3-vcstool python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# ---------- [4] Initialize rosdep ----------
RUN rosdep init || true && rosdep update

# ---------- [5] Create user ----------
RUN useradd -m ${USERNAME} && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ${USERNAME}
WORKDIR /home/${USERNAME}

# ---------- [6] Copy ROS2 Workspace ----------
COPY ../* /home/${USERNAME}/ros2_ws/src

# ---------- [7] Build ROS2 Workspace ----------
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd /home/${USERNAME}/ros2_ws && colcon build --symlink-install || true

# ---------- [8] Set Environment ----------
ENV ROS_DISTRO=${ROS_DISTRO}
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PYTHONPATH=/usr/lib/python3/dist-packages:/home/edgebot/ros2_ws/install/lib/python3.10/site-packages

# ---------- [9] Entrypoint ----------
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
CMD ["bash"]
