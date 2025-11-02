# ======================================================
# Edge Brain Base Image (Jetson or x86 GPU compatible)
# ======================================================
ARG ROS_DISTRO=humble
ARG CUDA_VERSION=12.2.0
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# ---------- System Setup ----------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git vim lsb-release gnupg2 build-essential cmake \
    python3 python3-pip python3-colcon-common-extensions \
    python3-rosdep python3-rosinstall python3-vcstool \
    locales && \
    locale-gen en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ---------- ROS2 Installation ----------
RUN echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -
RUN apt-get update && apt-get install -y ros-${ROS_DISTRO}-desktop && rm -rf /var/lib/apt/lists/*

# ---------- Python Dependencies ----------
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
RUN pip install numpy scipy matplotlib tqdm opencv-python \
    pybind11 tensorboard onnx onnxruntime \
    gymnasium==0.29.1 stable-baselines3==2.3.2

# ---------- ROS2 Workspace Setup ----------
ENV ROS_WS=/home/edgebot/ros2_ws
RUN mkdir -p ${ROS_WS}/src
WORKDIR ${ROS_WS}

# 如果你已有 repo，可直接复制进来：
# COPY ./edge-brain-go2 ./src/edge-brain-go2
# 或使用 vcs import:
# COPY ./ros2.repos ./
# RUN vcs import src < ros2.repos

# ---------- Build ----------
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --symlink-install

# ---------- Runtime Env ----------
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PYTHONPATH=/usr/lib/python3/dist-packages:${ROS_WS}/install/lib/python3.10/site-packages

# ---------- Default User ----------
RUN useradd -m edgebot && echo "edgebot ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER edgebot
WORKDIR /home/edgebot

# ---------- Entry Point ----------
COPY ./entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
CMD ["bash"]
