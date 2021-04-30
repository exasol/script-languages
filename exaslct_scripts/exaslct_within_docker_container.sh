#!/usr/bin/env bash

set -euo pipefail

RUNNER_IMAGE_NAME="$1"
shift 1

if [[ -t 1 ]]; then
  terminal_parameter=-it
else
  terminal_parameter=""
fi

quoted_arguments=''
for argument in "${@}"; do
  argument="${argument//\\/\\\\}"
  quoted_arguments="$quoted_arguments \"${argument//\"/\\\"}\""
done

RUN_COMMAND="/script-languages-container-tool/starter_scripts/exaslct_without_poetry.sh $quoted_arguments; RETURN_CODE=\$?; chown -R $(id -u):$(id -g) .build_output &> /dev/null; exit \$RETURN_CODE"

HOST_DOCKER_SOCKER_PATH="/var/run/docker.sock"
CONTAINER_DOCKER_SOCKER_PATH="/var/run/docker.sock"
DOCKER_SOCKET_MOUNT="$HOST_DOCKER_SOCKER_PATH:$CONTAINER_DOCKER_SOCKER_PATH"

docker run --rm $terminal_parameter -v "$PWD:$PWD" -v "$DOCKER_SOCKET_MOUNT" -w "$PWD" "$RUNNER_IMAGE_NAME" bash -c "$RUN_COMMAND"