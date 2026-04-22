SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

podman run -it \
  --device nvidia.com/gpu=all \
  -v "${SCRIPT_DIR}/../:/root/ros_ws/src/forest_segmentation" \
  -v "${SCRIPT_DIR}/../../forest_segmentation_data:/data" \
  -v "${SCRIPT_DIR}/../../forest_segmentation_logs:/logs" \
  -v "${SCRIPT_DIR}/.zsh_history:/root/.zsh_history" \
  forest_segmentation_devkit
