#!/bin/bash

set -e
set -u
set -o pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
IMAGE_NAME=exasol/script-languages-install-scripts-test-image-python-only
docker build -t $IMAGE_NAME -f "$SCRIPT_DIR/docker_python_only/Dockerfile" "$SCRIPT_DIR/.."

# Check if STDOUT is connected to a terminal. If so, we use Docker's interactive mode:
if [ -t 0 ]
then
   DOCKER_TTY_OPTION=-it
else
   DOCKER_TTY_OPTION=
fi

echo Use DOCKER_TTY_OPTION="$DOCKER_TTY_OPTION"

# shellcheck disable=SC2086
RUN_PIP_TESTS_EXECUTOR="docker run $DOCKER_TTY_OPTION -w /scripts/tests/install_scripts \"$IMAGE_NAME\"" PATH_TO_INSTALL_SCRIPTS="/scripts/install_scripts" bash install_scripts/run_pip_tests_with_epheramal_build_tools.sh --no-dry-run
