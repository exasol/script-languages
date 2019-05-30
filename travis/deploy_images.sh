#!/usr/bin/env bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
command="./exaslct push $($SCRIPT_DIR/deploy_docker_options.sh) $*"
echo "Executing command: $command"
bash -c "$command"