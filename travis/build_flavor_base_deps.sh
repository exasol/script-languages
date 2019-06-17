#!/usr/bin/env bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
command="./exaslct push $($SCRIPT_DIR/docker_options_build.sh) --build-name '$($SCRIPT_DIR/build_name.sh)' $FORCE_REBUILD --goal flavor_base_deps --force-rebuild-from flavor_base_deps $*"
echo "Executing command: $command"
bash -c "$command"