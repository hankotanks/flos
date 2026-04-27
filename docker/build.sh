SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

touch "${SCRIPT_DIR}/.zsh_history"
podman build -t forest_segmentation_devkit_2 -f "${SCRIPT_DIR}/ros2.Dockerfile" .