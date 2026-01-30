#!/bin/bash

ARCH=$(uname -m)

case "${ARCH}" in
    x86_64)
        PLATFORM="x86_64"
        ;;
    aarch64|arm64)
        # Jetson
        if [ -f /etc/nv_tegra_release ]; then
            PLATFORM="jetson"
        else
            # Standard ARM64
            PLATFORM="arm64"
        fi
        ;;
    *)
        echo "[ERROR] Unsupported architecture: ${ARCH}"
        exit 1
        ;;
esac

docker run -it --rm \
  --runtime nvidia \
  --net host \
  --pid host \
  --ipc host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  edge-brain-go2:${PLATFORM} 
