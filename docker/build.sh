SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

touch "${SCRIPT_DIR}/.zsh_history"
podman build -t flos_devkit -f "${SCRIPT_DIR}/Dockerfile" .