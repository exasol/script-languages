#!/bin/bash

set -e
set -u
set -o pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
IMAGE_NAME=exasol/script-languages-install-scripts-test-image
docker build -t $IMAGE_NAME -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR/.."

if [ -t 0 ]
then
   DOCKER_TTY_OPTION=-it
else
   DOCKER_TTY_OPTION=
fi

echo Use DOCKER_TTY_OPTION="$DOCKER_TTY_OPTION"

# shellcheck disable=SC2086
#docker run $DOCKER_TTY_OPTION -w /scripts/tests/install_scripts "$IMAGE_NAME"  bash run_apt_tests.sh --no-dry-run
# shellcheck disable=SC2086
RUN_PIP_TESTS_EXECUTOR="docker run $DOCKER_TTY_OPTION -w /scripts/tests/install_scripts \"$IMAGE_NAME\"" PATH_TO_INSTALL_SCRIPTS="/scripts/install_scripts" bash install_scripts/run_pip_tests.sh --no-dry-run
# shellcheck disable=SC2086
#docker run $DOCKER_TTY_OPTION -w /scripts/tests/install_scripts "$IMAGE_NAME"  bash run_r_remotes_tests.sh --no-dry-run
# shellcheck disable=SC2086
#docker run $DOCKER_TTY_OPTION -w /scripts/tests/install_scripts "$IMAGE_NAME"  bash run_ppa_tests.sh --no-dry-run
