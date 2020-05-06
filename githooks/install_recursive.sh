#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_DIR=$(git rev-parse --show-toplevel)
REPO_DIR="$(readlink -f "${REPO_DIR}")"

pushd $REPO_DIR &> /dev/null
echo
echo "Install hook for $REPO_DIR"
echo "=================================================================="
echo
bash "$REPO_DIR/githooks/install.sh"
echo
popd

submodule_paths=$(git submodule status --recursive | sed "s/^\s*//" | cut -f 2 -d " ")
for submodule_path in $submodule_paths
do
    submodule_path="$REPO_DIR/$submodule_path"
    submodule_path="$(readlink -f "${submodule_path}")"
    pushd "$submodule_path" &> /dev/null
    echo
    echo "Install hook for submodule $submodule_path"
    echo "==============================================================="
    echo
    install_script_for_submodule="githooks/install.sh"
    if [[ -f "$install_script_for_submodule" ]]
    then
        bash "$install_script_for_submodule"
    fi
    echo
done
