SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

touch "${SCRIPT_DIR}/.zsh_history"
podman build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t forest_segmentation_devkit -f "${SCRIPT_DIR}/Dockerfile" .