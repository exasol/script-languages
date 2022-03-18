#!/bin/bash
set -euo pipefail

echo Step One
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REBUILD=$2
FLAVOR=$1
ADDITIONAL_ARGUMENTS=""
if [ "$REBUILD" == "True" ]
then
  ADDITIONAL_ARGUMENTS="--force-rebuild"
fi
BUILD_PARAMETER="--no-shortcut-build"
SYSTEM_PARAMETER="--workers 7"

echo Step Two
#shellcheck source=./google-cloud-build/ci-scripts/scripts/generate_source_target_docker_options.sh
source "$SCRIPT_DIR/generate_source_target_docker_options.sh"

shift 2
generate_source_target_docker_options "$SCRIPT_DIR" "$@"

COMMAND="python3 -m exasol_script_languages_container_tool.main build --flavor-path "flavors/$FLAVOR" $BUILD_PARAMETER $ADDITIONAL_ARGUMENTS $SYSTEM_PARAMETER"
echo "Executing Command: $COMMAND"
$COMMAND


COMMAND="python3 -m exasol_script_languages_container_tool.main build-test-container $ADDITIONAL_ARGUMENTS $SYSTEM_PARAMETER"
echo "Executing Command: $COMMAND"
$COMMAND

echo
echo "=========================================================="
echo "Printing docker images"
echo "=========================================================="
echo
docker images | grep exa

# TODO use internal cache without commit hash as source and produce release target except for master branch builds
