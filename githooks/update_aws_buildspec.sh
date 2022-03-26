#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

echo -e "Updating AWS Buildspec."

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
ROOT_DIR="$SCRIPT_DIR/.."

pushd $ROOT_DIR > /dev/null
poetry run python -m script_languages_container_ci_setup.main generate-buildspecs --flavor-root-path "$ROOT_DIR/flavors"

git add "$ROOT_DIR/aws-code-build/ci"

popd > /dev/null

