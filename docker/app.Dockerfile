# docker/app.Dockerfile
# ======================================================
# edge-brain-go2 App 镜像
# ======================================================
# syntax=docker/dockerfile:1.5

FROM edge-brain-base:humble-cuda12.6

ARG USERNAME=edgebot
ENV ROS_DISTRO=humble
WORKDIR /home/${USERNAME}/ros2_ws

# ---------- 复制/拉取私有 repo ----------
# 使用 BuildKit + SSH 方式，不把凭证放入镜像
# 本地 clone 也可用 COPY 替代
RUN --mount=type=ssh \
    git clone git@github.com:your-org/edge-brain-go2.git src/

SHELL ["/bin/bash", "-c"]

# ---------- 构建自研 ROS2 包 ----------
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --packages-select edge_quadruped_controller --symlink-install || true

# ---------- 环境变量 ----------
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PYTHONPATH=/usr/lib/python3/dist-packages:/home/${USERNAME}/ros2_ws/install/lib/python3.10/site-packages

# ---------- 启动命令 ----------
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch edge_quadruped_controller bringup.launch.py"]
