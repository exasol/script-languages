#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

#shellcheck source=./scripts/build/setup_poetry_env.sh
source "$SCRIPT_DIR/setup_poetry_env.sh" "$@"

poetry run python3 -u "$SCRIPT_DIR/check_release.py"
