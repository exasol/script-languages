#!/bin/bash
set +x	
set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
#shellcheck source=./google-cloud-build/ci-scripts/scripts/generate_source_target_docker_options.sh
source "$SCRIPT_DIR/generate_source_target_docker_options.sh"

generate_source_target_docker_options "$SCRIPT_DIR" "$@"

PUSH_PARAMETER="--push-all --force-push"
SYSTEM_PARAMETER="--workers 7"

COMMAND="python3 -m exasol_script_languages_container_tool.main push --flavor-path "flavors/$FLAVOR" $SOURCE_OPTIONS $TARGET_OPTIONS $PUSH_PARAMETER $SYSTEM_PARAMETER"
echo "Executing Command: $COMMAND"
bash -c "$COMMAND"
echo
echo "=========================================================="
echo "Printing docker images"
echo "=========================================================="
echo
docker images | grep exa
