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

PUSH_PARAMETER="--push-all --force-push"
SYSTEM_PARAMETER="--workers 7"
if [ -f /workspace/build-status.txt ]
then
  rm /workspace/build-status.txt
fi
touch /workspace/build-status.txt
COMMAND="./exaslct push $FLAVOR_OPTIONS $SOURCE_OPTIONS $TARGET_OPTIONS $PUSH_PARAMETER $SYSTEM_PARAMETER"
echo "Executing Command: $COMMAND"
bash -c "$COMMAND" || echo "fail" >> /workspace/build-status.txt
echo
COMMAND="./exaslct push $FLAVOR_OPTIONS $SOURCE_OPTIONS $TARGET_OPTIONS $PUSH_PARAMETER $SYSTEM_PARAMETER"
echo "Executing Command: $COMMAND"
bash -c "$COMMAND" || echo "fail" >> /workspace/build-status.txt
echo
echo "=========================================================="
echo "Printing docker images"
echo "=========================================================="
echo
docker images | grep exa
