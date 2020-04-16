#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_DIR=$(git rev-parse --show-toplevel)

echo
echo "Install hook for $REPO_DIR"
echo "=================================================================="
echo
bash "$REPO_DIR/githooks/install.sh"
echo

submodule_paths=$(git submodule status --recursive | sed "s/^\s*//" | cut -f 2 -d " ")
for submodule_path in $submodule_paths
do
    echo
    echo "Install hook for submodule $REPO_DIR/$submodule_path"
    echo "==============================================================="
    echo
    install_script_for_submodule="$REPO_DIR/$submodule_path/githooks/install.sh"
    if [[ -f "$install_script_for_submodule" ]]
    then
        bash "$install_script_for_submodule"
    fi
    echo
done
