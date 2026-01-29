#!/bin/bash
docker run -it --rm \
  --runtime nvidia \
  --net host \
  --pid host \
  --ipc host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  edge-brain-go2
