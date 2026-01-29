#!/bin/bash
set -e

# ======================================================
# edge-brain-go2 Docker Build Script
# ======================================================

# 1️⃣ 配置参数
BASE_IMAGE_NAME=edge-brain-base
BASE_IMAGE_TAG=humble-cuda12.6
APP_IMAGE_NAME=edge-brain-go2
APP_IMAGE_TAG=latest

BASE_DOCKERFILE=docker/base.Dockerfile
APP_DOCKERFILE=docker/app.Dockerfile

# 2️⃣ 是否开启 BuildKit
export DOCKER_BUILDKIT=1

# 3️⃣ 构建 Base 镜像
echo "=============================="
echo "Building Base Image: ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}"
echo "=============================="
docker build -f ${BASE_DOCKERFILE} -t ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG} .

# 4️⃣ 构建 App 镜像
# 使用 SSH mount 来拉取私有 repo
echo "=============================="
echo "Building App Image: ${APP_IMAGE_NAME}:${APP_IMAGE_TAG}"
echo "=============================="
docker build \
    --ssh default \
    -f ${APP_DOCKERFILE} \
    -t ${APP_IMAGE_NAME}:${APP_IMAGE_TAG} .

echo "=============================="
echo "Docker Build Completed ✅"
echo "Base Image: ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}"
echo "App Image: ${APP_IMAGE_NAME}:${APP_IMAGE_TAG}"
echo "=============================="
