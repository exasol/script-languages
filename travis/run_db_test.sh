#!/usr/bin/env bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
command="./exaslct run-db-test $($SCRIPT_DIR/build_docker_options.sh) $*"
echo "Executing command: $command"
bash -c "$command"
