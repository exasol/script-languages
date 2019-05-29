#!/usr/bin/env bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
command="./exaslct export $($SCRIPT_DIR/docker_options.sh) $*"
echo "Executing command: $command"
bash -c "$command"