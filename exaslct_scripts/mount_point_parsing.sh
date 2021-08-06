#!/usr/bin/env bash

#set -e => immediately exit if any command [1] has a non-zero exit status
#set -u => reference to any variable you haven't previously defined is an error and causes the program to immediately exit.
#set -o pipefailt => This setting prevents errors in a pipeline from being masked.
#                    If any command in a pipeline fails,
#                    that return code will be used as the return code of the whole pipeline.
set -euo pipefail

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
    >&2 echo "Input directory $dir_path for parameter $current_arg does not exist."
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
    >&2 echo "Input file $file_path for parameter $current_arg does not exist."
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
    >&2 echo "INVALID ARGUMENT. Please adjust variable relevant_mount_point_arguments in $0!"
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
    #Better way to test if the argument is in $relevant_mount_point_arguments
    #would be to use [[ -v ...]], however this does not work correctly with bash 4.2
    if [[ -n "${relevant_mount_point_arguments[${current_arg}]-}" ]]; then
      _get_mount_point_path $current_arg $next_arg "${relevant_mount_point_arguments[${current_arg}]}"
    fi
  done
}

function _format_parameters() {
  result=""
  for param in "${@}"; do
    local found=0
    for check_param in "${!relevant_mount_point_arguments[@]}"; do
      if [[ $param == "${check_param}"=* ]]; then
        first_part="$(echo $param | cut -d= -f 1)"
        second_part="$(echo $param | cut -d= -f 2-)"
        result="$result $first_part $second_part"
        found=1
        break
      fi
    done
    if [[ $found -eq 0 ]]; then
      result="$result $param"
    fi
  done
  echo "$result"
}

declare -a mount_point_paths
formatted_params=$(_format_parameters "${@}")
_get_mount_point_paths $formatted_params

echo "${mount_point_paths[@]}"