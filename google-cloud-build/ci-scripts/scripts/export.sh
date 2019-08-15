#!/bin/bash
set +x	
set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source "$SCRIPT_DIR/generate_source_target_docker_options.sh"
source "$SCRIPT_DIR/generate_flavor_options.sh"

generate_flavor_options "$1"
shift 1
generate_source_target_docker_options $SCRIPT_DIR $*

touch /workspace/build-status.txt

COMMAND="./exaslct export $SOURCE_OPTIONS $TARGET_OPTIONS $FLAVOR_OPTIONS --workers 7"
echo "Executing Command: $COMMAND"
bash -c "$COMMAND" || echo "fail" > /workspace/build-status.txt
