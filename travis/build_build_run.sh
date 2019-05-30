#!/usr/bin/env bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
command="./exaslct push $($SCRIPT_DIR/build_docker_options.sh) $FORCE_REBUILD --goal build_run --force-rebuild-from build_run $*"
echo "Executing command: $command"
bash -c "$command"