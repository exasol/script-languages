#!/usr/bin/env bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
command="./exaslct push $($SCRIPT_DIR/docker_options_build.sh) $FORCE_REBUILD --goal release --force-rebuild-from release --force-rebuild-from flavor_customization $*"
echo "Executing command: $command"
bash -c "$command"