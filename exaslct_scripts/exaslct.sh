#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

EXASLCT_GIT_REF="9e08f15cd0a6ecdf27ff012817ec52272b10a7c0"
RUNNER_IMAGE_NAME="$(bash "$SCRIPT_DIR/construct_docker_runner_image_name.sh" "$EXASLCT_GIT_REF")"

bash $SCRIPT_DIR/exaslct_within_docker_container_without_container_build.sh "$RUNNER_IMAGE_NAME" "${@}"

