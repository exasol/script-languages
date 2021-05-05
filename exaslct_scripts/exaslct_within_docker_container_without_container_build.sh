#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

RUNNER_IMAGE_NAME="$1"
shift 1

FIND_IMAGE_LOCALLY=$(docker images -q "$RUNNER_IMAGE_NAME")
if [ -z "$FIND_IMAGE_LOCALLY" ]; then
  docker pull "$RUNNER_IMAGE_NAME"
fi

bash "$SCRIPT_DIR/exaslct_within_docker_container.sh" "$RUNNER_IMAGE_NAME" "${@}"
