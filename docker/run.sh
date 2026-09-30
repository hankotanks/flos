#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MOUNT_FLAGS=(
  -v "${SCRIPT_DIR}/../:/root/ros_ws/src/flos"
  -v "${SCRIPT_DIR}/.zsh_history:/root/.zsh_history"
)

if (( $# % 2 != 0 )); then
    echo "Usage: $0 [path_local path_container]..."
    exit 1
fi

while (( $# > 0 )); do
    path_local="$1"
    name="$2"
    shift 2
    mkdir -p "$path_local"
    MOUNT_FLAGS+=(-v "${path_local}:/root/ros_ws/${name}")
done

if [[ -n "${DISPLAY:-}" ]]; then
    MOUNT_FLAGS+=(--env "DISPLAY=$DISPLAY")
    MOUNT_FLAGS+=(--env "QT_X11_NO_MITSHM=1")
    MOUNT_FLAGS+=(--volume /tmp/.X11-unix:/tmp/.X11-unix:rw)
fi

podman run -it \
  --device nvidia.com/gpu=all \
  "${MOUNT_FLAGS[@]}" \
  --security-opt label=disable \
  flos_devkit
