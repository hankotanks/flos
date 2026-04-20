SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

podman run -it \
  --device nvidia.com/gpu=all \
  -v "${SCRIPT_DIR}/../:/home/dev/ros_ws/src/forest_segmentation":U \
  -v "${SCRIPT_DIR}/.zsh_history:/home/dev/.zsh_history":U \
  forest_segmentation_devkit
