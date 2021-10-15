#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

EXASLCT_GIT_REF="a800dabe4f1e52ddef27dad5473e9881bd83a2b8"
RUNNER_IMAGE_NAME="$(bash "$SCRIPT_DIR/construct_docker_runner_image_name.sh" "$EXASLCT_GIT_REF")"

bash $SCRIPT_DIR/exaslct_within_docker_container_without_container_build.sh "$RUNNER_IMAGE_NAME" "${@}"

