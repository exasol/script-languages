#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

echo -e "Updating starter scripts."

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
ROOT_DIR="$SCRIPT_DIR/.."

pushd "$ROOT_DIR" > /dev/null

poetry run python3 -m exasol_script_languages_container_tool.main install-starter-scripts --install-path "." --force-install
git add "./exaslct"
git add "./exaslct_scripts"

popd > /dev/null