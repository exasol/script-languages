#!/bin/bash

set -e
set -u
set -o pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
IMAGE_NAME=exasol/script-languages-install-scripts-test-image
docker build -t $IMAGE_NAME -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR/.."

docker run -it -w /scripts/tests/install_scripts $IMAGE_NAME  bash run_apt_tests.sh --no-dry-run
docker run -it -w /scripts/tests/install_scripts $IMAGE_NAME  bash run_pip_tests.sh --no-dry-run
docker run -it -w /scripts/tests/install_scripts $IMAGE_NAME  bash run_r_remotes_tests.sh --no-dry-run
