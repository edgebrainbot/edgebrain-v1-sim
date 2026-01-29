#!/bin/bash
set -e

# ======================================================
# edge-brain-go2 Docker Build Script
# - Auto ssh-agent handling
# - BuildKit enabled
# ======================================================

# -------- Image Names --------
BASE_IMAGE_NAME=edge-brain-base
BASE_IMAGE_TAG=humble-cuda12.6

APP_IMAGE_NAME=edge-brain-go2
APP_IMAGE_TAG=latest

# -------- Dockerfiles --------
BASE_DOCKERFILE=../dockerfile/base.Dockerfile
APP_DOCKERFILE=../dockerfile/app.Dockerfile

# -------- Enable BuildKit --------
export DOCKER_BUILDKIT=1

# -------- Start SSH agent --------
SSH_KEY="$HOME/.ssh/id_rsa"

if [ -f "$SSH_KEY" ]; then
    echo "[INFO] Starting ssh-agent and adding key $SSH_KEY"
    eval "$(ssh-agent -s)"
    ssh-add "$SSH_KEY"
    echo "[DEBUG] SSH_AUTH_SOCK=$SSH_AUTH_SOCK"
    
    # Fix socket permissions for Docker BuildKit
    if [ -n "$SSH_AUTH_SOCK" ] && [ -S "$SSH_AUTH_SOCK" ]; then
        chmod 666 "$SSH_AUTH_SOCK"
        echo "[DEBUG] Fixed SSH socket permissions"
    fi
    
    SSH_OPTION="--ssh default"
else
    echo "[INFO] No SSH key found at $SSH_KEY, building WITHOUT SSH"
    SSH_OPTION=""
fi

# ======================================================
# Build Base Image
# ======================================================
echo "=============================="
echo "Building Base Image: ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}"
echo "=============================="
docker build \
    -f ${BASE_DOCKERFILE} \
    -t ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG} .

# ======================================================
# Build App Image
# ======================================================
echo "=============================="
echo "Building App Image: ${APP_IMAGE_NAME}:${APP_IMAGE_TAG}"
echo "=============================="
docker build \
    ${SSH_OPTION} \
    -f ${APP_DOCKERFILE} \
    -t ${APP_IMAGE_NAME}:${APP_IMAGE_TAG} .

# ======================================================
# Done
# ======================================================
echo "=============================="
echo "Docker Build Completed ✅"
echo "Base Image: ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}"
echo "App Image:  ${APP_IMAGE_NAME}:${APP_IMAGE_TAG}"
echo "=============================="
