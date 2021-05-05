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

function create_env_file() {
  touch "$tmpfile_env"
  if [ -n "${TARGET_DOCKER_PASSWORD-}" ]; then
    echo "TARGET_DOCKER_PASSWORD=$TARGET_DOCKER_PASSWORD" >> "$tmpfile_env"
  fi
  if [ -n "${SOURCE_DOCKER_PASSWORD-}" ]; then
    echo "SOURCE_DOCKER_PASSWORD=$SOURCE_DOCKER_PASSWORD" >> "$tmpfile_env"
  fi
}

function create_env_file_debug_protected() {
  shell_options="$-"
  case $shell_options in
  *x*) set +x ;;
  *) echo &>/dev/null ;;
  esac

  create_env_file "$1"

  case $shell_options in
  *x*) set -x ;;
  *) echo &>/dev/null ;;
  esac
}

old_umask=$(umask)
umask 277
tmpfile_env=$(mktemp)
trap 'rm -f -- "$tmpfile_env"' INT TERM HUP EXIT

create_env_file_debug_protected "$tmpfile_env"

docker run --env-file "$tmpfile_env" --rm $terminal_parameter -v "$PWD:$PWD" -v "$DOCKER_SOCKET_MOUNT" -w "$PWD" "$RUNNER_IMAGE_NAME" bash -c "$RUN_COMMAND"

umask "$old_umask"
