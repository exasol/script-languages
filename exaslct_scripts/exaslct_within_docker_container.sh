#!/usr/bin/env bash

#####################################################################################
###REMEMBER TO KEEP THIS FILE IN SYNC WITH exaslct_within_docker_container_slim.sh!!!
#####################################################################################

#set -e => immediately exit if any command [1] has a non-zero exit status
#set -u => reference to any variable you haven't previously defined is an error and causes the program to immediately exit.
#set -o pipefailt => This setting prevents errors in a pipeline from being masked.
#                    If any command in a pipeline fails,
#                    that return code will be used as the return code of the whole pipeline.
set -euo pipefail

RUNNER_IMAGE_NAME="$1"
shift 1

if [[ -t 1 ]]; then
  terminal_parameter=-it
else
  terminal_parameter=""
fi

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
declare -a mount_point_paths
mount_point_paths=($(bash "$SCRIPT_DIR"/mount_point_parsing.sh "${@}"))

quoted_arguments=''
for argument in "${@}"; do
  argument="${argument//\\/\\\\}"
  quoted_arguments="$quoted_arguments \"${argument//\"/\\\"}\""
done

#After finalizing docker run we need to change owner (current user) of all directories which were mounted again
#For that we call chown with a list of the respective directories
#In order to avoid syntax errors we need to encapsulate all those directories with quotes here
chown_directories=''
for mount_point in "${mount_point_paths[@]}"; do
  mount_point="${mount_point//\\/\\\\}"
  chown_directories="$chown_directories \"${mount_point//\"/\\\"}\""
done

chown_directories_cmd=''
if [[ -n "$chown_directories" ]]; then
  chown_directories_cmd="chown -R $(id -u):$(id -g) $chown_directories;"
fi

#For all mount pounts (directories in argument list) we need
# 1. For the host argument: Resolve relative paths and resolve symbolic links
# 2. For the container argument: Resolve relative paths, but keep symbolic links
mount_point_parameter=''
for mount_point in "${mount_point_paths[@]}"; do
  host_dir_name=$(readlink -f "${mount_point}")
  container_dir_name=$(realpath -s "${mount_point}")
  mount_point_parameter="$mount_point_parameter-v ${host_dir_name}:${container_dir_name} "
done

# Still need to "CHOWN" .build_output
# because it is a default value for --output-path, and hence might not be part of $chown_directories
RUN_COMMAND="/script-languages-container-tool/starter_scripts/exaslct_without_poetry.sh $quoted_arguments; RETURN_CODE=\$?; $chown_directories_cmd chown -R $(id -u):$(id -g) .build_output &> /dev/null; exit \$RETURN_CODE"

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
docker run --network host --env-file "$tmpfile_env" --rm $terminal_parameter -v "$PWD:$PWD" -v "$DOCKER_SOCKET_MOUNT" -w "$PWD" ${mount_point_parameter[@]} "$RUNNER_IMAGE_NAME" bash -c "$RUN_COMMAND"

umask "$old_umask"
