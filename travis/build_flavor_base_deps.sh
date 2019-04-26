#!/usr/bin/env bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
command="./exaslct push $($SCRIPT_DIR/docker_options.sh) $FORCE_REBUILD --goal flavor_base_deps --force-rebuild-from flavor_base_deps $*"
echo "Executing command: $command"
bash -c "$command"