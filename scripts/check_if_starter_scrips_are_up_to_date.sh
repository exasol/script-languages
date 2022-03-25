#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

pushd "$SCRIPT_DIR/.." 2>/dev/null
#bash "./githooks/update_exaslct_starter_scripts.sh"
popd 2>/dev/null
git status --porcelain=v1 -uno
git diff --cached; git diff --cached --summary;
[ -z "$(git status --porcelain=v1 -uno 2>/dev/null)" ]

