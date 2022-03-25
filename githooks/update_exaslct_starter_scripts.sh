#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

echo -e "Updating starter scripts."

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
ROOT_DIR="$SCRIPT_DIR/.."

poetry run python3 -m exasol_script_languages_container_tool.main install-starter-scripts --install-path "$ROOT_DIR" --force-install
git add "$ROOT_DIR/exaslct"
git add "$ROOT_DIR/exaslct_scripts"