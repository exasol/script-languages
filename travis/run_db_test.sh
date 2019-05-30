#!/usr/bin/env bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
command="./exaslct run-db-test $($SCRIPT_DIR/docker_options_build.sh) $*"
echo "Executing command: $command"
bash -c "$command"
