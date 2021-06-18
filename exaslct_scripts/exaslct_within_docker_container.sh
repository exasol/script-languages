#!/usr/bin/env bash

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

declare -A relevant_mount_point_arguments
relevant_mount_point_arguments["--flavor-path"]=in_path
relevant_mount_point_arguments["--export-path"]=out_path
relevant_mount_point_arguments["--output-directory"]=out_path
relevant_mount_point_arguments["--temporary-base-directory"]=out_path
relevant_mount_point_arguments["--cache-directory"]=in_path
relevant_mount_point_arguments["--save-directory"]=out_path
relevant_mount_point_arguments["--task-dependencies-dot-file"]=out_file
#relevant_mount_point_arguments["--test-folder"]=in_path # TODO reactive in #37
#relevant_mount_point_arguments["--test-file"]=in_file # TODO reactive in #37

function _get_mount_point_path_for_in_dir() {
  local current_arg=$1
  local dir_path=$2

  if [[ -d $dir_path ]]; then
    mount_point_paths+=("$dir_path")
  else
    echo "Input directory $dir_path for parameter $current_arg does not exist."
    exit 1
  fi
}

function _get_mount_point_path_for_in_file() {
  local current_arg=$1
  local file_path=$2
  if [[ -f $file_path ]]; then
    local rel_dir_name=''
    rel_dir_name="$(dirname "${file_path}")"
    mount_point_paths+=("$rel_dir_name")
  else
    echo "Input file $file_path for parameter $current_arg does not exist."
    exit 1
  fi
}

function _get_mount_point_path_for_out_path() {
  local dir_path=$1
  #Create out directories if necessary
  if [[ ! -d $dir_path ]]; then
    mkdir -p "$dir_path"
  fi
  mount_point_paths+=("$dir_path")
}

function _get_mount_point_path_for_out_file() {
  local file_path=$1
  local rel_dir_name=''
  rel_dir_name="$(dirname "${file_path}")"
  #Create out directories if necessary
  if [[ ! -d $file_path ]]; then
    mkdir -p "$rel_dir_name"
  fi
  mount_point_paths+=("$rel_dir_name")
}

function _get_mount_point_path() {
  local current_arg=$1
  local next_arg=$2
  local arg_type=$3

  case $arg_type in
  in_path)
    _get_mount_point_path_for_in_dir "${current_arg}" "${next_arg}"
    ;;
  in_file)
    _get_mount_point_path_for_in_file "${current_arg}" "${next_arg}"
    ;;
  out_path)
    _get_mount_point_path_for_out_path "${next_arg}"
    ;;
  out_file)
    _get_mount_point_path_for_out_file "${next_arg}"
    ;;
  *)
    echo "INVALID ARGUMENT. Please adjust variable relevant_mount_point_arguments in $0!"
    exit 1
    ;;
  esac
}

function _get_mount_point_paths() {
  local lenArgs="$#"
  for ((idxArg = 1; idxArg < lenArgs; idxArg++)); do
    current_arg=${!idxArg}
    next_arg_idx=$((idxArg + 1))
    next_arg=${!next_arg_idx}
    if [ -v relevant_mount_point_arguments[$current_arg] ]; then
      _get_mount_point_path $current_arg $next_arg ${relevant_mount_point_arguments[$current_arg]}
    fi
  done
}

declare -a mount_point_paths
_get_mount_point_paths "${@}"

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
RUN_COMMAND="/script-languages-container-tool/starter_scripts/exaslct_without_poetry.sh $quoted_arguments; RETURN_CODE=\$?; chown -R $(id -u):$(id -g) $chown_directories; chown -R $(id -u):$(id -g) .build_output &> /dev/null; exit \$RETURN_CODE"

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
