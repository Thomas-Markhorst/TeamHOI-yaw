#!/usr/bin/env bash
set -euo pipefail
container_name=teamhoi-yaw-container
if docker container inspect "$container_name" >/dev/null 2>&1; then
    docker start "$container_name"
else
    docker run -d --init --name "$container_name" --gpus all --shm-size=10G \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
      -v /fs01/tud/tcmarkhorst/projects/TeamHOI-yaw:/workspace/TeamHOI-yaw \
      -v /projB/tud/tcmarkhorst/data:/workspace/data \
      -w /workspace/TeamHOI-yaw teamhoi-yaw:latest
fi
